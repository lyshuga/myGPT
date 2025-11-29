import os

# class TextDataset:
#     def __init__(self, B, T,  split):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         assert split in ["train", "val"]

#         with open('input.txt', 'r', encoding='utf-8') as f:
#             text = f.read()
        
        

class DataLoaderLite:

    def __init__(self, B, T, process_rank, num_processes, split):

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"]

        data_root = 'edu_fineweb10B'

        shards = os.listdir(data_root)
        shards = [s for s in shards]

        #TODO

    def reset(self):
        pass

    def next_batch(self):
        pass