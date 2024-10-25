import numpy as np
import torch
import os
def gen_random_index(size,select_length):
    random_sample_0 = np.random.randint(size[0], size=(select_length, 1))
    for s in range(1,len(size)):
        random_sample = np.random.randint(size[s], size=(select_length, 1))
        random_sample_0=np.hstack((random_sample_0,random_sample))

    return random_sample_0

def random_subset(net,th,jobname,dataset,model_num,eps):

    model = net

    # th = 0.1

    save_weights = {}
    for name, param in model.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad = False
        else:
            if 'weight' in name :
                p = param
                p_size = p.size()
                lengeth = len(p.reshape(-1, 1))
                select_len = int(th * lengeth)
                index = gen_random_index(p_size, select_len)
                save_weights[name] = index
    PATH = './save_weights/'+dataset+'/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    np.save(PATH+ jobname+'_eps_'+str(eps)+'_'+str(model_num)+'_random_'+str(th)+'_index.npy', save_weights, allow_pickle=True)
    print("save to ",'./save_weights/'+dataset+'/'+ jobname+'_eps_'+str(eps)+'_'+str(model_num)+'_random_'+str(th)+'_index.npy')
    return './save_weights/'+dataset+'/'+ jobname+'_eps_'+str(eps)+'_'+str(model_num)+'_random_'+str(th)+'_index.npy'

def switch_weights(new_weights, net, epoch,file):
    # save_dict=np.load("save_weights.npy",allow_pickle=True).item()
    # save_dict = np.load("save_grad_index.npy", allow_pickle=True).item()
    if epoch <= 200:
        save_dict = np.load(file, allow_pickle=True).item()
        # save_dict = np.load("save_weights_th1.npy", allow_pickle=True).item()
    else:
        save_dict = np.load(file, allow_pickle=True).item()
    # save_dict = np.load("save_weights_random_3.npy", allow_pickle=True).item()
    with torch.no_grad():
        for name, param in net.named_parameters():
            if 'cls_token' in name or 'pos_embed' in name:
                param.requires_grad = False
            else:
                if 'weight' in name:

                    index_name = name
                    index = save_dict[index_name]
                    new_para = new_weights[name]
                    print(index_name)
                    if 'head.' in name:
                        param = new_para
                    for i in index:
                        if len(i)==1:
                            param[i]=new_para[i]
                        if len(i)==2:
                            f=i[0]
                            s=i[1]

                            param[f][s]=new_para[f][s]

                        if len(i)==3:
                            f = i[0]
                            s = i[1]
                            t= i[2]
                            param[f][s][t] = new_para[f][s][t]
                        if len(i)==4:
                            f = i[0]
                            s = i[1]
                            t= i[2]
                            f4=i[3]
                            param[f][s][t][f4] = new_para[f][s][t][f4]

    return net


def ST_selection(net,num):
    th_list=np.load('./save_weights/ST/threshold.npy',allow_pickle=True)
    th=th_list[num]
    model=net
    save_weights = {}
    for name, param in model.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad = False
        else:
            if 'head.' in name:
                pass
            else:
                op = param.detach().cpu().numpy()
                # p = param.detach().numpy()
                index = np.argwhere(abs(op) > th)
                save_weights[name] = index


    np.save('./save_weights/ST/save_weights_'+str(num)+'.npy', save_weights, allow_pickle=True)
    print("Save")
    return './save_weights/ST/save_weights_'+str(num)+'.npy'

