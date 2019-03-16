import torch

if torch.cuda.is_available():
    print('gpu supported')
else:
    print('gpu unsupported')

print(torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))

t = torch.FloatTensor([2,3])
print(t)

print(t.cuda(0))
