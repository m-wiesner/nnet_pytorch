import torch
import torch.nn as nn
import torch.nn.functional as F
from .Energy import ClassificationEnergy, LFMMIEnergy, TargetEnergy
from samplers.Sampler import Sampler


class EnergyLoss(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--energy-type',
            choices=['LFMMIEnergy', 'ClassificationEnergy'],
            help='Which Energy function to use',
            default='LFMMIEnergy',
        )
        Sampler.add_args(parser)
        energy_args, extra = parser.parse_known_args()
        energy_type = eval(energy_args.energy_type) 
        energy_type.add_args(parser) 

    @classmethod
    def build_objective(cls, conf):
        energy_class = eval(conf.get('energy_type', 'LFMMIEnergy'))
        energy = energy_class.build_energy(conf)
        sampler = Sampler.build_sampler(conf)
        return EnergyLoss(energy, sampler)
           
    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return {
            'sampler': Sampler.add_state_dict(
                s1['sampler'], s2['sampler'], fraction, iteration=iteration, 
            )
        }
    
    def __init__(self, energy, sampler,):
        super(EnergyLoss, self).__init__()
        self.energy = energy
        self.sampler = sampler
         
    def forward(self, model, sample, precomputed=None):
        # Use precomputed values to avoid extra forward pass through model
        if precomputed is not None:
            x = precomputed
        else:
            x = model(sample)[0]
      
        B = x.size(0)
        T = x.size(1)
       
        # Sample speech from model distribution 
        generated_samples, num_iters = self.sampler.generate_like(
            sample, model, self.energy, precomputed=x
        )
        
        # We need the data energy
        data_energy = self.energy(model, sample, precomputed=x)
        
        # We need the expected energy of the current model
        # \nabla Loss = E(x) - \mathbb{E}_{p_{\theta}\left(x\right)}[ E(x) ]
        expected_energy = self.energy(
            model,
            generated_samples,
            precomputed=model(generated_samples)[0]
        )
        
        # We need to minimize a loss instead of maximize p(x) so we negate the
        # gradient
        loss_ebm = -(expected_energy - data_energy) / (B * T)
        print("Expected_E: ", expected_energy.data.item(), "E: ", data_energy.data.item())
        return loss_ebm, None

    def state_dict(self):
        return {
            'sampler': self.sampler.state_dict(),
        }      
  
    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict['sampler'])

    def decorrupt(self, model, sample, num_steps=None, targets=None):
        if num_steps is None:
            num_steps = self.sampler.max_steps
        self.sampler.max_steps = 1
        self.sampler.min_steps = 1
        self.sampler.stop_crit = 'fixed'

        # Decorrupt using p(x) and so we have to sample from the marginal.
        # If we decorrupt to a fixed target we sample from p(x|y) and so the
        # the generate_type is set to prior. 
        if targets is None:
            self.sampler.sampling_type = 'marg'
        else:
            self.sampler.sampling_type = 'prior'

        yield sample.input 
        for i in range(num_steps): 
            yield self.sampler.generate_like(
                sample, model, self.energy, precomputed=None,
                generate_type='decorrupt', targets=targets
            )[0].input

    def generate_from_buffer(self):
        return self.sampler.buffer

