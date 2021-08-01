# Copyright 2021
# Apache 2.0

import torch

from functools import partial
from collections import namedtuple
from copy import deepcopy
import random

from objectives.Energy import LFMMIEnergy, ClassificationEnergy, TargetEnergy
from .optim.SGLD import SGLD
from .optim.AdamSGLD import AdamSGLD
from .optim.AcceleratedSGLD import AcceleratedSGLD


class Sampler(torch.nn.Module):
    '''
        This is and Energy-based Sampler. It is responsible for generating
        samples using an underlying neural network and an Energy function
        defined on these network outputs.  
    '''    
    Minibatch = namedtuple('Minibatch', ['input', 'metadata']) 

    @staticmethod
    def add_args(parser):
        parser.add_argument('--sgld-min-steps', type=int, default=20)
        parser.add_argument('--sgld-max-steps', type=int, default=20)
        parser.add_argument('--sgld-buffer-size', type=str, default="(10000, 250, 100)") # B x T x D
        parser.add_argument('--sgld-reinit-p', type=float, default=0.05)
        parser.add_argument('--sgld-thresh', type=float, default=0.0)
        parser.add_argument('--sgld-clip', type=float, default=1.0)
        parser.add_argument('--sgld-init-val', type=float, default=1.0)
        parser.add_argument('--sgld-debug', action='store_true')
        parser.add_argument('--sgld-stop-crit',
            choices=[
                'fixed',
                'uniform_rand',
                'gauss_rand',
                'target',
                'dynamic',
            ],
            help='The stopping criteria for the SGLD updates.'
                '   fixed        -- stop after (min_steps + max_steps)/2 '
                '                   iterations.'
                '   uniform_rand -- stop after uniform(min_steps, max_steps) '
                '                   iterations.'
                '   gauss_rand   -- stop after number of steps sampled from '
                '                   gaussian with: '
                '                   mu = (min_steps + max_steps)/2'
                '                   var = (max_steps - min_steps) / 2'
                '   target       -- stop within threshold of target value'
                '   dynamic      -- change number of steps based on '
                '                   difference between E[E(x)] and E(x).',
            default='fixed',
        ) 
        parser.add_argument('--sgld-optim', type=str,
            choices=['SGLD', 'AcceleratedSGLD', 'AdamSGLD'],
            default='SGLD',
            help='The type of optimizer used during SGLD.'
        )
        parser.add_argument(
            '--sgld-sampling-type',
            choices=['marg', 'post', 'prior'],
            help='The type of sampling used to compute the energy function '
                'maximized during sampling. '
                '  marg -- the targets are marginalized. The energy '
                '          function is the logsumexp. We sample x from p(x).'
                '  post -- the targets, y, are sampled from p(y|x). We then '
                '          sample x from p(x|y). '
                '  prior -- the targets, y, are sampled from p(y). p(y) is '
                '          approximated from data. This is a file that has '
                '          each utterance followed by the corresponding '
                '          sequence of y, where each y is some integer from a '
                '          finite vocabulary. We sample x from p(x|y).',
            default='marg',
        )
        parser.add_argument(
            '--sgld-priors',
            default=None,
            help='The path to the targets for priors based sampling, i.e., '
                ' args.sgld_sampling_type == "prior".'
        )
        # Parse the sgld optim arg in order to add the approprate additional 
        # sgld optim dependent arguments
        args_sgld_optim, extra = parser.parse_known_args()
        sgld_optim_class = eval(args_sgld_optim.sgld_optim)
        sgld_optim_class.add_args(parser)

    @classmethod
    def build_sampler(cls, conf):
        sgld_optim_class = eval(conf['sgld_optim'])
        sgld_optim = sgld_optim_class.build_partial(conf) 
        sgld_priors = None

        # Check that the priors are specified when using the prior
        # sampling type
        if conf.get('sgld_sampling_type', 'marg') == 'prior':
            if conf.get('sgld_priors', None) is None:
                raise RuntimeError('Using sampling type <prior> requires '
                    'specifying a file with priors using '
                    '                                    '
                    '    --sgld-priors <FILENAME>.       '
                )
            else:
                sgld_priors = conf.get('sgld_priors', None)

        # Check that the sgld stopping criterion is compatible with the 
        # sampling type
        if (
            conf.get('sgld_stop_crit', 'fixed') == 'target' and
            conf.get('sgld_sampling_type', 'marg') in ('prior', 'post')
        ):
            raise RuntimeError('sgld_stop_crit == "target" is not supported '
                'with post or prior sampling types.'
             )
       
        datasets = eval(conf['datasets'])
        max_len = 0
        for d in datasets:
            widths = d['min_chunk_width'], d['left_context'], d['right_context']
            max_len = max(max_len, sum(widths))
            max_len = max(max_len, d['chunk_width'])
        
        buffer_size = eval(conf.get('sgld_buffer_size', "(10000, 250, 100)"))
        if max_len > buffer_size[1]:
            raise RuntimeError("The data length is longer than the buffer") 
        
        # Returned the sampler
        return Sampler(
            sgld_buffer_size=buffer_size,
            sgld_reinit_p=conf.get('sgld_reinit_p', 0.05),
            sgld_init_val=conf.get('sgld_init_val', 1.0),
            sgld_min_steps=conf.get('sgld_min_steps', 20),
            sgld_max_steps=conf.get('sgld_max_steps', 20),
            sgld_sampling_type=conf.get('sgld_sampling_type', 'marg'),
            sgld_stop_crit=conf.get('sgld_stop_crit', 'fixed'),
            sgld_thresh=conf.get('sgld_thresh', 0.0),
            sgld_clip=conf.get('sgld_clip', 1.0), 
            sgld_debug=conf.get('sgld_debug', True),
            sgld_priors=sgld_priors,
            sgld_optim=sgld_optim,
        )

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        s1 = deepcopy(s1)
        buffsize = len(s2['buffer'])
        if 'buffer' not in s1 or len(s1['buffer']) == 0:
            s1['buffer'] = s2['buffer']
            s1['buffer_numsteps'] = s2['buffer_numsteps']
            return s1
        num_samples = int(fraction * buffsize)
        idxs = torch.randperm(buffsize)
        idxs = idxs[:num_samples]
        if iteration is not None:
            start_idx = iteration * num_samples
            s1['buffer'][start_idx:start_idx + num_samples] = s2['buffer'][idxs]
            s1['buffer_numsteps'][start_idx: start_idx + num_samples] = s2['buffer_numsteps'][idxs]
        else:
            s1['buffer'][idxs] = s2['buffer'][idxs]
            s1['buffer_numsteps'][idxs] = s2['buffer_numsteps'][idxs]
        return s1
    
    def __init__(self,
        sgld_buffer_size=(10000, 250, 100),
        sgld_reinit_p=0.05,
        sgld_init_val=1.0,
        sgld_min_steps=20,
        sgld_max_steps=20,
        sgld_thresh=0.0,
        sgld_stop_crit='fixed',
        sgld_clip=1.0,
        sgld_debug=True,
        sgld_sampling_type='marg',
        sgld_priors=None,
        sgld_optim=SGLD,
    ):
        super(Sampler, self).__init__()
        self.buffer_size = sgld_buffer_size
        self.register_buffer(
            'buffer',
            torch.zeros(sgld_buffer_size).uniform_(-sgld_init_val, sgld_init_val)
        )
        self.register_buffer('buffer_numsteps', torch.zeros(sgld_buffer_size[0]))
        self.reinit_p = sgld_reinit_p
        self.init_val = sgld_init_val
        self.min_steps = sgld_min_steps
        self.max_steps = sgld_max_steps
        self.thresh = sgld_thresh
        self.stop_crit = sgld_stop_crit
        self.clip = sgld_clip
        self.debug = sgld_debug
        self.sampling_type = sgld_sampling_type
        self.priors = sgld_priors
        self.optim = sgld_optim
        if self.stop_crit == 'fixed':
            self.steps = (self.min_steps + self.max_steps) // 2
        else:
            self.steps = self.min_steps

        if self.sampling_type == 'prior':
            self.priors_offsets = self.load_priors()
     
    def generate_like(self, sample, model, energy,
        data_energy=None, generate_type='pd', targets=None,
        normalize=False, 
    ):
        '''
            Generate samples using the sampler to have the same dimensions,
            type, and device as x. model and energy are used to produce the 
            scores of the sample under the model. data_energy is a target
            energy value that can be used in optimization to help guide the
            sampler toward producing better samples quickly, especially when
            using the AcceleratedSGLD optimizer or the the target stopping
            criterion.

            generate_type:
                pd          -- persistent divergence
                buffer      -- from buffer
                target      -- to target
                decorrupt   -- use inputs 
        '''
        # Some initial checks. Initializing the buffer if it is still empty
        # This should never happen at this point ...
        if self.buffer is None or len(self.buffer) == 0:
            self.init_buffer_like(sample.input) 
        
        B = sample.input.size(0)
        # Initialize synthetic sample
        if generate_type == 'pd':
            sample_synthetic, metadata = self.init_pd_like(sample) 
            numsteps = metadata['buffer_numsteps']
        elif generate_type == 'target':
            sample_synthetic, numsteps = self.init_rand_like(sample)
            metadata = sample.metadata
        elif generate_type == 'decorrupt':
            sample_synthetic = sample.input
            metadata = sample.metadata 
            numsteps = torch.zeros(B).to(sample.input.device)
        
        # Determine the energy function to use
        if self.sampling_type == 'prior':
            energy_ = TargetEnergy()
            if targets is None and hasattr(sample, 'target'):
                T = sample.target.size(1)
                targets = self.sample_targets(B, T)
            energy_fun = partial(
                energy_.forward, model, target=targets, precomputed=None, normalize=normalize,
            )
        elif self.sampling_type == 'post':
            raise NotImplementedError('<post> sampling is not yet implemented')
        elif self.sampling_type == 'marg': 
            energy_fun = partial(energy, model, precomputed=None, normalize=normalize)
       
        # Treat inputs as updated parameters and track gradients
        sample_synthetic.requires_grad = True
        sample_synthetic_params = torch.nn.Parameter(sample_synthetic)
        optim = self.optim([sample_synthetic_params], finalval=data_energy)

        # Create a stopping criterion based on num sgld iters taken and current
        # value of the function
        if self.stop_crit in ('fixed', 'dynamic'):
            stop_crit = lambda *args: args[0] >= self.steps
        elif self.stop_crit == 'uniform_rand':
            stop_crit = lambda *args: args[0] >= self.min_steps + \
                int((random.random() - 1e-09) * (self.max_steps - self.min_setps))
        elif self.stop_crit == 'gauss_rand':
            mu = (self.max_steps + self.min_steps) / 2
            var = (self.max_steps - self.min_steps) / 2
            stop_crit = lambda *args: args[0] >= int(
                min(
                    max(self.min_steps, random.gauss(mu, var)),
                    self.max_steps,
                )
            )
        elif self.stop_crit == 'target':
            stop_crit = lambda *args: (
                    (
                (args[1] - data_energy) / abs(data_energy) <= self.thresh and
                args[0] >= self.min_steps
                    ) or args[0] >= self.max_steps
            ) 
        
        # Set up synthetic minibatch and evaluate the energy
        y = energy_fun(Sampler.Minibatch(sample_synthetic_params, metadata))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [sample_synthetic_params], self.clip
        )
        k = 0 # The iteration counter
        if self.debug:
            logging_stats = {
                'init': 
                    {
                        'steps': numsteps.mean(),
                        'num_new': (numsteps == 0).sum().item(),
                        'E_data': data_energy,  
                    },
                'stats': [],
            }
        # Main Loop
        while not stop_crit(k, y.data.item()):
            sample_synthetic_params.grad = torch.autograd.grad(
                y, [sample_synthetic_params], retain_graph=False
            )[0].clone()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [sample_synthetic_params], self.clip
            )
            if self.debug:
                logging_stats['stats'].append(
                    {
                        'E': y.data.item(),
                        'std': sample_synthetic_params.std().data.item(),
                        'mean': sample_synthetic_params.mean().data.item(),
                        'grad_norm': grad_norm.data.item(),
                    }
            )
            numsteps += 1
            optim.step(numsteps=numsteps, startval=y.data.item())
            optim.zero_grad()
            y = energy_fun(Sampler.Minibatch(sample_synthetic_params, metadata))  
            k += 1 
        
        if self.debug:
            logging_stats['stats'].append(
                {
                    'E': y.data.item(),
                    'std': sample_synthetic_params.std().data.item(),
                    'mean': sample_synthetic_params.mean().data.item(),
                    'grad_norm': grad_norm.data.item(),
                }
            ) 
            self.generation_print(logging_stats)  
        
        x = sample_synthetic_params.detach() 
        if len(self.buffer) > 0 and generate_type == 'pd':
            buffer_idxs = metadata['buffer_idxs']
            self.buffer[buffer_idxs, :x.size(1), :x.size(2)] = x
            self.buffer_numsteps[buffer_idxs] = numsteps
        return Sampler.Minibatch(x, metadata), k

    def generation_print(self, logging_stats):
        print()
        print('--------------------- Sampling -----------------------------')
        print('init: ', end='')
        for k, v in sorted(logging_stats['init'].items(), key=lambda x: x[0]):
            print(f'{k}: {v}', end=' --- ')
        print()
        for i, stats in enumerate(logging_stats['stats']):
            print(f'k: {i}', end='   ')
            for k, v in sorted(stats.items(), key=lambda x: x[0]):
                print(f'{k}: {v}', end='   ')
            print()
        Sampled_Energy = logging_stats['stats'][-1]['E']
        Target_Energy = logging_stats['init']['E_data']
        print('-------------------------------------------------------------')
        print(f"Sampled_Energy: {Sampled_Energy} Target_Energy {Target_Energy}")
        print('-------------------------------------------------------------')

    def init_buffer_like(self, data):
        '''
            Initialize the buffer with examples that have the same shape as
            data.
        '''
        bs = self.buffer_size[0]
        cw, dim = data.size(1), data.size(2)
        self.buffer = torch.FloatTensor(
            bs, cw, dim
        ).uniform_(-self.init_val, self.init_val).to(data.device)
    
    def init_random_like(self, data):
        '''
            Return a tensor of size data.size() with uniform random values
            betwen -self.init_val and self.init_val.
        '''
        bs, cw, dim = data.size(0), data.size(1), data.size(2)
        samples = torch.FloatTensor(
            bs, cw, dim
        ).uniform_(-self.init_val, self.init_val).to(data.device)
        numsteps = torch.zeros(bs).to(data.device)
        return samples, numsteps

    def init_from_buffer_like(self, data):
        '''
            Return a tensor of size data.size() with values sampled from the
            buffer.
        '''
        idxs = torch.randint(0, len(self.buffer), (data.size(0),)).to(data.device)
        T, D = data.size(1), data.size(2)
        if D > self.buffer.size(2):
            raise RuntimeError("The data dimension is larger than what is in the buffer")
        return self.buffer[idxs, :T, :D], self.buffer_numsteps[idxs], idxs

    def init_pd_like(self, sample):
        '''
            Return a tensor of size data.size() with values drawn randomly, but
            also from the buffer.
        '''
        x = sample.input
        metadata = sample.metadata 
        buff_samples, buff_numsteps, idxs = self.init_from_buffer_like(x)
        rand_samples, rand_numsteps = self.init_random_like(x)
        
        choose_rand = (torch.rand(x.size(0)) < self.reinit_p).float()[:, None, None].to(x.device)
        samples = choose_rand * rand_samples + (1 - choose_rand) * buff_samples 
        numsteps = choose_rand.view(-1) * rand_numsteps + (1 - choose_rand.view(-1)) * buff_numsteps 
        
        metadata.update(
            {
                'buffer_idxs': idxs,
                'buffer_numsteps': numsteps
            }
        ) 
        return samples, metadata 

    def load_priors(self):
        '''
            Loads the offsets in file, self.priors, for each utterance to
            support fast sampling of subsequences from this file when sampling
            according to p(y), where y is a target sequence.
        '''
        if self.priors is None:
            raise ValueError(f'filename {fname} cannot be None')
        
        tgt_lines = {}
        with open(self.priors) as f:
            i = 0
            start = f.tell()
            line = f.readline()
            while line:
                # length is -1 because the first token is the uttid
                tgt_lines[i] = (start, len(line.split())-1)
                start = f.tell()
                line = f.readline()
                i += 1
        return tgt_lines
    
    def sample_targets(self, bs, length):
        '''
            Samples target sequences from priors file.  
        '''
        f = open(self.priors)
        offsets = random.choices(self.priors_offsets, k=bs)
        tgts = []
        for offset in offsets:
            idx = 1 + int((random.random() - 1e-09) * (offset[1] - (length-1)))
            f.seek(offset[0])
            # The first token is the uttid
            tgts.append(
                [int(pdf) for pdf in f.readline().split()[idx: idx + length]]
            )
        f.close()
        return tgts

    def update(self, expected_energy, energy):
        if self.stop_crit == 'dynamic':
            if expected_energy < energy + self.thresh: 
                self.steps = max(self.min_steps, self.steps - 1)
            else:
                self.steps = min(self.max_steps, self.steps + 1)

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({'steps': self.steps})
        return state_dict

    def load_state_dict(self, state_dict):
        steps = state_dict.pop('steps', None)
        if steps is not None:
            self.steps = steps 
        super().load_state_dict(state_dict)
