#!/usr/bin/perl

################################################################################
#
# This script takes two text files in Kaldi transcription-format, and identifies
# lines in the second file that are also seen in the first.
#
@ARGV == 2 || die "Usage: $0 train.txt test.txt > seen_utts.txt";
#
# The basic purpose is to identify whether there is significant overlap between
# the traning and test data by checking what fraction of utterances are not OOV
#
# The lines in both the input files are assumed to have the format
#
#     UTTERANCE_ID w1 w2 w3 ... wN
#
# The output file is essentially the same as the second file, with a count for
# each line in the file, i.e.
#
#     NN UTTERANCE_ID w1 w2 w3 ... wN
#
# where NN is how often the utterance "w1 w2 ... wN" is seen in the first file
#
# A value of -1 for NN indicates that the line is not in the expected format
# A value of  0 indicates an unseen utterance
# A value greater than 0 indicates the training count of a duplicate utterance
#
################################################################################

$train = shift @ARGV;
$test  = shift @ARGV;


# Read and store the training utterances

open (TRAIN, "< $train") || die "$0: Unable to open training text $train\n";
my $nLines = 0;
my %SeenInTraining = ();
my $line = "";
while (<TRAIN>) {
    $nLines++;
    if ($_ =~ m:(\S+)\s+(\S+.*):) {
	$line = $2;
	if (exists $SeenInTraining{$line}) {
	    $SeenInTraining{$line} += 1;
	} else {
	    $SeenInTraining{$line} = 1;
	}
    } else {
	print STDERR "\tSkipping line $nLines in $train:\n\tnot formatted as UTT_ID TRANSCRIPT\n";
    }
}
print STDERR "Finished reading $nLines lines from $train\n";
close (TRAIN);


# Read and check the multiplicity of the test utterances

open (TEST, "< $test") || die "$0: Unable to open test text $test\n";
$nLines = 0;
my %SeenInTest = ();
while (<TEST>) {
    $nLines++;
    if ($_ =~ m:(\S+)\s+(\S+.*):) {
	$line = $2;
	if (exists $SeenInTest{$line}) {
	    $SeenInTest{$line} += 1;
	} else {
	    $SeenInTest{$line} = 1;
	}
	print STDOUT sprintf ("%d\t%s", (exists $SeenInTraining{$line}) ? $SeenInTraining{$line} : 0, $_);
    } else {
	print STDERR "\tSkipping line $nLines in $test:\n\tnot formatted as UTT_ID TRANSCRIPT\n";
	print STDOUT sprintf ("%d\t%s", -1, $_);
    }
}
print STDERR "Finished processing $nLines lines from $test\n";
close (TEST);

# Generate the Good-Turing estimate of seeing a training utterance repeated in test

$nLines = 0;
my $nSingletons = 0;
my $estimatedRepetitionRate = 0.0;
foreach $line (keys %SeenInTraining) {
    $nLines += $SeenInTraining{$line}; # Count of this line in training data
    $nSinglteons++ if ($SeenInTraining{$line} == 1);
}
$estimatedRepetitionRate = 1.0 - ($nSinglteons/$nLines) unless ($nLines == 0);

# Generate some statistics

$nLines = 0;
my $nSeenLines = 0;
my $expectedDuplicate = 0;
foreach $line (keys %SeenInTest) {
    $nLines += $SeenInTest{$line}; # Count of this line in test data
    if (exists $SeenInTraining{$line}) {
	$nSeenLines += $SeenInTest{$line}; # All these test lines are seen/duplicate
	if ($SeenInTraining{$line} > 1) {
	    # These test lines are duplicated in training as well
	    $expectedDuplicate += $SeenInTest{$line};
	}
    }
}
# print STDERR sprintf ("%8d lines in $test\n", $nLines);
print STDERR sprintf ("%8d lines (%.1f\%) are expected to have been seen, per Good-Turing on $train\n",
		      $estimatedRepetitionRate*$nLines, 100*$estimatedRepetitionRate);
print STDERR sprintf ("%8d lines (%.1f\%) are actually seen in $test\n",
		      $nSeenLines, ($nLines>0) ? (100*$nSeenLines/$nLines) : 0);
print STDERR sprintf ("%8d of the seen lines (%.1f\%) are also seen more than once in $train\n",
		      $expectedDuplicate, ($nSeenLines>0) ? (100*$expectedDuplicate/$nSeenLines) : 0);
exit (0);
