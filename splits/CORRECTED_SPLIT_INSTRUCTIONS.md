# Place the corrected split file here as 'corrected_split.txt'.
#
# The corrected split is identical to the official ICBHI split except that
# the 12 contaminated test-side recordings of patients 156 and 218 are
# REMOVED entirely (they are no longer in either partition). The training
# partition is unchanged.
#
# Specifically, REMOVE these lines (compared with the official split):
#
#   156_*                test
#   218_*                test
#
# (Keep the train-side lines for patients 156 and 218 untouched.)
#
# After dropping in the corrected split, delete this README to avoid
# confusion.
