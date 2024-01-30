
from common.load_data_3dhp_mae import Fusion
from common.opt import opts
import torch
import torch.distributed as dist

opt = opts().parse()

# print('opt', opt)

train_data = Fusion(opt=opt,
                            train=True,
                            #dataset=dataset,
                            root_path='/media/arthur/Data/Human-Posture-Reconstruction/data/')
# train_sampler = torch.utils.data.distributed.DistributedSampler(
#             train_data, num_replicas=torch.distributed.get_world_size(), rank=dist.get_rank())
train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=opt.batch_size,
            num_workers=int(opt.workers),
            pin_memory=True
            )

test_data = Fusion(opt=opt,
                           train=False,
                           #dataset=dataset,
                           root_path=opt.root_path)
test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            pin_memory=True)
        
print('test data', test_data)
print('test dataloader', test_dataloader)
