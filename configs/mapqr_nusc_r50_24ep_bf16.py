_base_ = './mapqr_nusc_r50_24ep.py'
fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=8, workers_per_gpu=8)
evaluation = dict(interval=999)
