
# from torchvision.transforms.functional import resize
# import torch
# a=torch.ones(2,2,10,10)
# # b=torch.zeros(2,1,5,5)
# a[:,:,1:6,2:7]=2
# # b[:,:,2:,2:]=3
# print(a)
# # print(b)
# # print(a.mul(b))
# # print(a.reshape(-1,5,5))
# #
# c=resize(a,[5,5])
# # a=torch.resize_as_(a,b)
# # print(a)
# print(c)
# import torch
# a= torch.Tensor([
#     [4,1,2,0,0],
#     [2,4,0,0,0],
#     [1,1,1,6,5],
#     [1,2,2,2,2],
#     [3,0,0,0,0],
#     [2,2,0,0,0]])
# index = torch.LongTensor([[3],[2],[5],[5],[1],[2]])
# print(a.size(),index.size())
# b = torch.gather(a, 1,index-1)
# print(b)
# import torch
# a= torch.Tensor([
#     [0.4,0.1,0.2,0.,0.3],
#     [0.2,0.4,0.2,0.1,0.1],
#     [0.1,0.1,0.1,0.6,0],
#     [0.1,0.3,0.2,0.2,0.2],
#     [0.3,0.0,0.7,0,0],
#     [0.2,0.2,0.0,0.5,0.1]])
# index = torch.LongTensor([[1],[2],[4],[2],[3],[4]])
# print(a.size(),index.size())
# b = torch.gather(a, 1,index-1)
# print(b)
# import torch
# targets=torch.zeros(3,5)
# scr1=torch.Tensor([[0.1],[0.2],[0.3]])
# scr2=torch.Tensor([[0.6],[0.5],[0.4]])
# index1 = torch.LongTensor([[3],[2],[5]])
# index2 = torch.LongTensor([[1],[2],[4]])
# targets.scatter_(1,index1-1,scr1)
# targets.scatter_(1,index2-1,scr2)
# print(targets)