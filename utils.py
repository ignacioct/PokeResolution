import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from IPython.display import clear_output


# Callback to track training
class TrackTraining(Callback):

    # Add metrics to track to dictionary
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            
    # Update local metrics and plots
    def on_epoch_end(self, epoch, logs={}):
        # Store metrics from log. Check for new metrics.
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Get metrics (no validation metrics) 
        metrics = [x for x in logs if 'val' not in x]
        
        # Plot metrics and clear output
        clear_output(wait=True)

        for _, metric in enumerate(metrics):
            # Plot training metric
            plt.plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label='train_' + metric)
            # Plot validation metric
            if logs['val_' + metric]:
                plt.plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)

            plt.legend()
            plt.grid()
            plt.xticks(range(1, epoch + 2))

        plt.tight_layout()
        plt.show()