----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- loss functions:
--   + negative-log likelihood, using log-normalized output units (SoftMax)
--   + mean-square error
--   + margin loss (SVM-like)
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- 10-class problem
noutputs = 10

----------------------------------------------------------------------
print '==> define loss'

-- The loss works like the MultiMarginCriterion: it takes
-- a vector of classes, and the index of the grountruth class
-- as arguments.

criterion = nn.ClassNLLCriterion()
----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
