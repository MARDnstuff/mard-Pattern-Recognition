from config.logging import setUpLogging
from data.creditApproval import CreditApproval
from data.redditBitcoinSentiment import RedditBitcoinSentiment
from data.iris import Iris
from validation.k_fold_cross_validation import k_fold_cv
from validation.hold_out import hold_out
from validation.leave_one_out_cross_validation import leave_one_out
import logging


# Logging
setUpLogging()
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    # dt = CreditApproval()
    dt = RedditBitcoinSentiment()
    # dt = Iris()
    # k_fold_cv(dt)
    # hold_out(dt)
    leave_one_out(dt)

