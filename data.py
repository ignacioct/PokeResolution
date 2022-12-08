import splitfolders
from tensorflow.keras.preprocessing import image_dataset_from_directory


class Images:
    def __init__(self,
                 path,
                 split_path,
                 splits=(.8, .1, .1)):
        self.path = path
        self.split_path = split_path
        assert sum(splits) == 1
        self.splits = splits
    
    # Return datasets for a split
    def _get_split(self, split):
        return image_dataset_from_directory(
            self.split_path+f'{split}/hr',
            seed=123,
            label_mode=None,
            shuffle=True,
            batch_size=None
        )

    # Generate splits and return all datasets
    def get_high_res_partitions(self, createFolders=True):
        # Create folder splits if needed
        if createFolders:
            splitfolders.ratio(self.path, output=self.split_path, seed=254, ratio=self.splits)

        # Load dataset from folder splits
        train_ds = self._get_split("train")
        val_ds = self._get_split("val")
        test_ds = self._get_split("test")
        
        return train_ds, val_ds, test_ds

