#!/usr/bin/perl

################################################################################
#
# Analyse the output of identify_seen_utts.pl to identify any conversations that
# contain a high percentage of seen utterances.
#
# The script takes the STDOUT output of identify_seen_utts.pl as input, and
# writes out the fraction of seen utterances in each conversation.
#
# WARNING: See the hard-coded assumption that the utterance IDs have the format
#
#               SpeakerID_ConversationID_UtteranceNumber
#
# Specifically, the script assumes that
#
#    -- IF at least one underscore is present in the Kaldi utterance ID, then
#       everything after the LAST UNDERSCORE is the utterance number.
#    -- IF at least two underscores are present in the Kaldi utterance ID, then
#       everything before the FIRST UNDERSCORE is the speaker ID.
#
################################################################################


my $count = 0;
my $convID = "";
my %nUtts = ();
my %nReps = ();
my $nLines = 0;

while (<>) {
    next unless ($count, $convID) = $_ =~ m:^(\d+)\s+(\S+)\s+\S*:; # Skip lines that don't have the expected format
    $nLines++;
    $convID =~ s:(\S+)\_.*:$1:; # Strip off the utterance number if present
    $convID =~ s:[^\_]+\_(\S+):$1:; # Strip off the speaker number if present

    if (exists $nUtts{$convID}) {
	$nUtts{$convID} += 1;
	$nReps{$convID} += 1 if ($count>0);
    } else {
	$nUtts{$convID} = 1;
	$nReps{$convID} = ($count>0) ? 1 : 0;
    }
}
printf STDERR ("Processed %d lines from STDIN\n", $nLines); 
foreach $convID (sort keys %nUtts) {
    printf STDOUT ("%s\t%5.1f\t(%3d of %3d)\n",
		   $convID, (100*$nReps{$convID}/$nUtts{$convID}), $nReps{$convID}, $nUtts{$convID});
}

exit (0);
