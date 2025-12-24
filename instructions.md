You are working with an indoor radar dataset. Each sample in data/training_set is stored as a .mat.pt file containing a tensor with:

6 input channels (indices 0-5): Different radar heatmaps showing signal strength
- Index 0: Static range-azimuth heatmap
- Index 1: Dynamic range-azimuth heatmap
- Index 2: Static range-elevation heatmap
- Index 3: Dynamic range-elevation heatmap
- Index 4: Static range-velocity heatmap
- Index 5: Dynamic range-velocity heatmap
1 target channel (index 6): Semantic label map with classes:
- -1: Background (no target)
- 0: Suitcase
- 1: Chair
- 2: Human
- 3: Wall The final goal is to predict the semantic label map for new samples.

Your tasks are:

- Create train/validation split (80/20) from training_set
- Design one CNN for semantic segmentation with Weighted CrossEntropyLoss as the loss function:
  - Use reasonable weights for background vs other classes
  - Document why you’ve chosen these weights in `ce_weight_selection.md`
- Design the CNN architecture:
  - Decide on the number of convolutional layers
  - Determine the number of output classes and dimensions
  - Choose training hyperparameters:
    - Number of epochs to train
    - Learning rate
  - Report architectural choices and reasoning to `architecture_reasoning.md`
- Train the model:
  - Track training and validation loss for each epoch
  - Create a comparison visualization saved to `accuracy_comparison.png` (e.g., bar chart of per-class accuracies between training and validation sets or loss curves)
  - Evaluate on the validation set and calculate per-class accuracy
  - Assign the following variables:
    - `weighted_ce_class_accuracies`: dict mapping class labels to accuracy (float 0-100)
    - `weighted_ce_final_val_loss`: float, final validation loss for weighted CE
  - Write model performance analysis to `performance_analysis.md`
