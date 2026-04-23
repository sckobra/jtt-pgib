import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from Configures import data_args, train_args, model_args, mcts_args
from my_mcts import mcts
from tqdm import tqdm
from proto_join import join_prototypes_by_activations
from utils import PlotUtils
from torch_geometric.utils import to_networkx
from itertools import accumulate
from torch_geometric.datasets import MoleculeNet
import pdb
import random
from sklearn.metrics import roc_auc_score



def warm_only(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = False


def joint(model):
    for p in model.model.gnn_layers.parameters():
        p.requires_grad = True
    model.model.prototype_vectors.requires_grad = True
    for p in model.model.last_layer.parameters():
        p.requires_grad = True


def append_record(info):
    f = open('./log/hyper_search.txt', 'a')
    f.write(info)
    f.write('\n')
    f.close()

def train_GC_first_pass(model_type):
    print('start loading data====================')
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    dataloader = get_dataloader(dataset, data_args.dataset_name, train_args.batch_size, data_split_ratio=data_args.data_split_ratio) # train, val, test dataloader 나눔

    print('training FIRST PASS===============')

    gnnNets = GnnNets(input_dim, output_dim, model_args) 
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]

    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print("Dataset : ", data_args.dataset_name)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices 

    best_acc = 0.0

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()

            # prototype projection
            for i in range( gnnNets.model.prototype_vectors.shape[0] ): 
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i*10, len(data_indices)): 
                    data = dataset[data_indices[j]] 
                    if data.y == label: 
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i]) 
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break


            # prototype merge
            share = True
            if train_args.share: 
                if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)) :  
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,  dataloader['train'], optimizer)

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        for i, batch in enumerate(dataloader['train']):
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch) 

            loss = criterion(logits, batch.y)

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device) 
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            #diversity loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[: ,i])))
            prototype_numbers = accumulate(prototype_numbers)
            n = 0
            ld = 0

            for k in prototype_numbers:    
                p = gnnNets.model.prototype_vectors[n : k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3 
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device) 
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2)) 

            #CHANGE LOSS FUNCTION IN SECOND FUNCTION: if in spurious set, add higher loss for that sample
            if model_args.cont:
                loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 
            else:
                loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            #acc checks if prediction equal to batch at y (if is correct)
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())


        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()))


        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion) 
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

        test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
        fid_pos, fid_neg, charact = compute_fidelity(dataloader['test'], gnnNets)
        worst_acc, group_accs = compute_worst_group_accuracy(dataloader['test'], gnnNets)
        group_accs_str = ", ".join(f"class {k}: {v:.4f}" for k, v in sorted(group_accs.items()))
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f} | AUC: {test_state['auc']:.3f}")
        print(f"  Fidelity+ : {fid_pos:.4f} | Fidelity-: {fid_neg:.4f} | Charact: {charact:.4f}")
        print(f"  Worst-Group Acc: {worst_acc:.4f} | Per-Class: [{group_accs_str}]")
        append_record("Test epoch {:2d}, loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, fid+: {:.4f}, fid-: {:.4f}, charact: {:.4f}, worst_group_acc: {:.4f}, per_class: {}".format(
            epoch, test_state['loss'], test_state['acc'], test_state['auc'],
            fid_pos, fid_neg, charact, worst_acc, group_accs))

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)


    print(f"The best validation accuracy is {best_acc}.")

    # report test msg
    gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best.pth')) # .to_device()
    gnnNets.to_device()
    gnnNets.eval()

    annotated_data = []

    #USE BATCH.TODATALIST, variables in order for accuracy, just map
    for split in ['train', 'eval', 'test']:
        for batch in dataloader[split]:
            batch = batch.to(model_args.device)
            # Standard forward pass
            with torch.no_grad():
                if model_args.cont:
                    logits, *_ = gnnNets(batch)
                else:
                    logits, *_ = gnnNets(batch)
            
            _, prediction = torch.max(logits, -1)
            
            # Convert batch to list of individual objects
            data_list = batch.to_data_list()
            
            for i, data in enumerate(data_list):
                # True if correct, False if wrong
                data.spurious = (prediction[i] == batch.y[i]).item()
                annotated_data.append(data)

    test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
    print(f"Test | Dataset: {data_args.dataset_name:s} | model: {model_args.model_name:s}_{model_type:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
    append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))

    return test_state['acc'], annotated_data, dataset

def train_GC(model_type):

    print('start loading data====================')
    __, annotated_data, dataset = train_GC_first_pass(model_type)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    dataloader = get_dataloader(annotated_data, data_args.dataset_name, train_args.batch_size, data_split_ratio=data_args.data_split_ratio) # train, val, test dataloader 나눔

    print('training SECOND PASS====================')

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
    gnnNets.to_device()
    criterion = nn.CrossEntropyLoss(reduction='none') # Return loss for each sample (for training)
    eval_criterion = nn.CrossEntropyLoss()            # Scalar loss for eval/test
    optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    avg_nodes = 0.0
    avg_edge_index = 0.0
    for i in range(len(dataset)):
        avg_nodes += dataset[i].x.shape[0]
        avg_edge_index += dataset[i].edge_index.shape[1]

    avg_nodes /= len(dataset)
    avg_edge_index /= len(dataset)
    print("Dataset : ", data_args.dataset_name)
    print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

    best_acc = 0.0
    data_size = len(dataset)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
        os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

    early_stop_count = 0
    data_indices = dataloader['train'].dataset.indices 

    best_acc = 0.0

    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        ld_loss_list = []

        if epoch >= train_args.proj_epochs and epoch % 50 == 0:
            gnnNets.eval()

            # prototype projection
            for i in range( gnnNets.model.prototype_vectors.shape[0] ): 
                count = 0
                best_similarity = 0
                label = gnnNets.model.prototype_class_identity[0].max(0)[1]
                for j in range(i*10, len(data_indices)): 
                    data = dataset[data_indices[j]] 
                    if data.y == label: 
                        count += 1
                        coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i]) 
                        if similarity > best_similarity:
                            best_similarity = similarity
                            proj_prot = prot
                    if count >= train_args.count:
                        gnnNets.model.prototype_vectors.data[i] = proj_prot
                        print('Projection of prototype completed')
                        break


            # prototype merge
            share = True
            if train_args.share: 
                if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)) :  
                    join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,  dataloader['train'], optimizer)

        gnnNets.train()
        if epoch < train_args.warm_epochs:
            warm_only(gnnNets)
        else:
            joint(gnnNets)

        #USE BATCH.TODATALIST, variables in order for accuracy, just map
        for i, batch in enumerate(dataloader['train']):
            if model_args.cont:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
            else:
                logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch) 

            raw_loss = criterion(logits, batch.y) 
            spurious_mask = batch.spurious.float()
            per_sample_loss = raw_loss + (spurious_mask * 0.5)
            loss = per_sample_loss.mean()
            

            if model_args.cont:
                prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device) 
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                positive_sim_matrix = sim_matrix * prototypes_of_correct_class
                negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

                contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
                contrastive_loss = - torch.log(contrastive_loss).mean()

            #diversity loss
            prototype_numbers = []
            for i in range(gnnNets.model.prototype_class_identity.shape[1]):
                prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[: ,i])))
            prototype_numbers = accumulate(prototype_numbers)
            n = 0
            ld = 0

            for k in prototype_numbers:    
                p = gnnNets.model.prototype_vectors[n : k]
                n = k
                p = F.normalize(p, p=2, dim=1)
                matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3 
                matrix2 = torch.zeros(matrix1.shape).to(model_args.device) 
                ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2)) 

            #CHANGE LOSS FUNCTION IN SECOND FUNCTION: if in spurious set, add higher loss for that sample
            #use contrastive loss - better performance
            #test diff loss values - hyperparameter sweep 
            if model_args.cont:
                loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 
            else:
                loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            #acc checks if prediction equal to batch at y (if is correct)
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            ld_loss_list.append(ld.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())


        # report train msg
        print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
              f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
        append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()))


        # report eval msg
        eval_state = evaluate_GC(dataloader['eval'], gnnNets, eval_criterion)
        print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
        append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

        test_state, _, _ = test_GC(dataloader['test'], gnnNets, eval_criterion)
        fid_pos, fid_neg, charact = compute_fidelity(dataloader['test'], gnnNets)
        worst_acc, group_accs = compute_worst_group_accuracy(dataloader['test'], gnnNets)
        group_accs_str = ", ".join(f"class {k}: {v:.4f}" for k, v in sorted(group_accs.items()))
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f} | AUC: {test_state['auc']:.3f}")
        print(f"  Fidelity+: {fid_pos:.4f} | Fidelity-: {fid_neg:.4f} | Charact: {charact:.4f}")
        print(f"  Worst-Group Acc: {worst_acc:.4f} | Per-Class: [{group_accs_str}]")
        append_record("Test epoch {:2d}, loss: {:.3f}, acc: {:.3f}, auc: {:.3f}, fid+: {:.4f}, fid-: {:.4f}, charact: {:.4f}, worst_group_acc: {:.4f}, per_class: {}".format(
            epoch, test_state['loss'], test_state['acc'], test_state['auc'],
            fid_pos, fid_neg, charact, worst_acc, group_accs))

        # only save the best model
        is_best = (eval_state['acc'] > best_acc)

        if eval_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = eval_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)


    print(f"The best validation accuracy is {best_acc}.")

    # report test msg
    gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best.pth')) # .to_device()
    gnnNets.to_device()
    test_state, _, _ = test_GC(dataloader['test'], gnnNets, eval_criterion)
    print(f"Test | Dataset: {data_args.dataset_name:s} | model: {model_args.model_name:s}_{model_type:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f} | AUC: {test_state['auc']:.3f}")
    append_record("loss: {:.3f}, acc: {:.3f}, auc: {:.3f}".format(test_state['loss'], test_state['acc'], test_state['auc']))

    # Fidelity evaluation
    fid_pos, fid_neg, charact = compute_fidelity(dataloader['test'], gnnNets)
    print(f"Fidelity+ (necessity): {fid_pos:.4f} | Fidelity- (sufficiency): {fid_neg:.4f} | Charact: {charact:.4f}")
    append_record(f"fid+: {fid_pos:.4f}, fid-: {fid_neg:.4f}, charact: {charact:.4f}")

    # Worst-group accuracy
    worst_acc, group_accs = compute_worst_group_accuracy(dataloader['test'], gnnNets)
    group_accs_str = ", ".join(f"class {k}: {v:.4f}" for k, v in sorted(group_accs.items()))
    print(f"Worst-Group Accuracy: {worst_acc:.4f} | Per-Class: [{group_accs_str}]")
    append_record(f"worst_group_acc: {worst_acc:.4f}, per_class: {group_accs}")

    return test_state['acc']



# train for graph classification
# def train_GC(model_type):
    
#     print('start loading data====================')
#     dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
#     input_dim = dataset.num_node_features
#     output_dim = int(dataset.num_classes)

#     dataloader = get_dataloader(dataset, data_args.dataset_name, train_args.batch_size, data_split_ratio=data_args.data_split_ratio) # train, val, test dataloader 나눔

#     print('start training model==================')

#     gnnNets = GnnNets(input_dim, output_dim, model_args) 
#     ckpt_dir = f"./checkpoint/{data_args.dataset_name}/"
#     gnnNets.to_device()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = Adam(gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

#     avg_nodes = 0.0
#     avg_edge_index = 0.0
#     for i in range(len(dataset)):
#         avg_nodes += dataset[i].x.shape[0]
#         avg_edge_index += dataset[i].edge_index.shape[1]

#     avg_nodes /= len(dataset)
#     avg_edge_index /= len(dataset)
#     print("Dataset : ", data_args.dataset_name)
#     print(f"graphs {len(dataset)}, avg_nodes{avg_nodes :.4f}, avg_edge_index_{avg_edge_index/2 :.4f}")

#     best_acc = 0.0
#     data_size = len(dataset)

#     # save path for model
#     if not os.path.isdir('checkpoint'):
#         os.mkdir('checkpoint')
#     if not os.path.isdir(os.path.join('checkpoint', data_args.dataset_name)):
#         os.mkdir(os.path.join('checkpoint', f"{data_args.dataset_name}"))

#     early_stop_count = 0
#     data_indices = dataloader['train'].dataset.indices 

#     best_acc = 0.0

#     for epoch in range(train_args.max_epochs):
#         acc = []
#         loss_list = []
#         ld_loss_list = []

#         if epoch >= train_args.proj_epochs and epoch % 50 == 0:
#             gnnNets.eval()

#             # prototype projection
#             for i in range( gnnNets.model.prototype_vectors.shape[0] ): 
#                 count = 0
#                 best_similarity = 0
#                 label = gnnNets.model.prototype_class_identity[0].max(0)[1]
#                 for j in range(i*10, len(data_indices)): 
#                     data = dataset[data_indices[j]] 
#                     if data.y == label: 
#                         count += 1
#                         coalition, similarity, prot = mcts(data, gnnNets, gnnNets.model.prototype_vectors[i]) 
#                         if similarity > best_similarity:
#                             best_similarity = similarity
#                             proj_prot = prot
#                     if count >= train_args.count:
#                         gnnNets.model.prototype_vectors.data[i] = proj_prot
#                         print('Projection of prototype completed')
#                         break


#             # prototype merge
#             share = True
#             if train_args.share: 
#                 if gnnNets.model.prototype_vectors.shape[0] > round(output_dim * model_args.num_prototypes_per_class * (1-train_args.merge_p)) :  
#                     join_info = join_prototypes_by_activations(gnnNets, train_args.proto_percnetile,  dataloader['train'], optimizer)

#         gnnNets.train()
#         if epoch < train_args.warm_epochs:
#             warm_only(gnnNets)
#         else:
#             joint(gnnNets)

#         for i, batch in enumerate(dataloader['train']):
#             if model_args.cont:
#                 logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, sim_matrix, _ = gnnNets(batch)
#             else:
#                 logits, probs, active_node_index, graph_emb, KL_Loss, connectivity_loss, prototype_pred_loss, _ = gnnNets(batch) 

#             loss = criterion(logits, batch.y)

#             if model_args.cont:
#                 prototypes_of_correct_class = torch.t(gnnNets.model.prototype_class_identity[:, batch.y]).to(model_args.device) 
#                 prototypes_of_wrong_class = 1 - prototypes_of_correct_class
#                 positive_sim_matrix = sim_matrix * prototypes_of_correct_class
#                 negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

#                 contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
#                 contrastive_loss = - torch.log(contrastive_loss).mean()

#             #diversity loss
#             prototype_numbers = []
#             for i in range(gnnNets.model.prototype_class_identity.shape[1]):
#                 prototype_numbers.append(int(torch.count_nonzero(gnnNets.model.prototype_class_identity[: ,i])))
#             prototype_numbers = accumulate(prototype_numbers)
#             n = 0
#             ld = 0

#             for k in prototype_numbers:    
#                 p = gnnNets.model.prototype_vectors[n : k]
#                 n = k
#                 p = F.normalize(p, p=2, dim=1)
#                 matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]).to(model_args.device) - 0.3 
#                 matrix2 = torch.zeros(matrix1.shape).to(model_args.device) 
#                 ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2)) 

#             if model_args.cont:
#                 loss = loss + train_args.alpha2 * contrastive_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 
#             else:
#                 loss = loss + train_args.alpha2 * prototype_pred_loss + model_args.con_weight*connectivity_loss + train_args.alpha1 * KL_Loss 

#             # optimization
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_value_(gnnNets.parameters(), clip_value=2.0)
#             optimizer.step()

#             ## record
#             _, prediction = torch.max(logits, -1)
#             loss_list.append(loss.item())
#             ld_loss_list.append(ld.item())
#             acc.append(prediction.eq(batch.y).cpu().numpy())


#         # report train msg
#         print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | Ld: {np.average(ld_loss_list):.3f} | "
#               f"Acc: {np.concatenate(acc, axis=0).mean():.3f}")
#         append_record("Epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, np.average(loss_list), np.concatenate(acc, axis=0).mean()))


#         # report eval msg
#         eval_state = evaluate_GC(dataloader['eval'], gnnNets, criterion)
#         print(f"Eval Epoch: {epoch} | Loss: {eval_state['loss']:.3f} | Acc: {eval_state['acc']:.3f}")
#         append_record("Eval epoch {:2d}, loss: {:.3f}, acc: {:.3f}".format(epoch, eval_state['loss'], eval_state['acc']))

#         test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
#         print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")           

#         # only save the best model
#         is_best = (eval_state['acc'] > best_acc)

#         if eval_state['acc'] > best_acc:
#             early_stop_count = 0
#         else:
#             early_stop_count += 1

#         if early_stop_count > train_args.early_stopping:
#             break

#         if is_best:
#             best_acc = eval_state['acc']
#             early_stop_count = 0
#         if is_best or epoch % train_args.save_epoch == 0:
#             save_best(ckpt_dir, epoch, gnnNets, model_args.model_name, eval_state['acc'], is_best)


#     print(f"The best validation accuracy is {best_acc}.")
    
#     # report test msg
#     gnnNets = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_{model_type}_{model_args.readout}_best.pth')) # .to_device()
#     gnnNets.to_device()
#     test_state, _, _ = test_GC(dataloader['test'], gnnNets, criterion)
#     print(f"Test | Dataset: {data_args.dataset_name:s} | model: {model_args.model_name:s}_{model_type:s} | Loss: {test_state['loss']:.3f} | Acc: {test_state['acc']:.3f}")
#     append_record("loss: {:.3f}, acc: {:.3f}".format(test_state['loss'], test_state['acc']))

#     # Fidelity evaluation
#     fid_pos, fid_neg, charact = compute_fidelity(dataloader['test'], gnnNets)
#     print(f"Fidelity+ (necessity): {fid_pos:.4f} | Fidelity- (sufficiency): {fid_neg:.4f} | Charact: {charact:.4f}")
#     append_record(f"fid+: {fid_pos:.4f}, fid-: {fid_neg:.4f}, charact: {charact:.4f}")

#     # Worst-group accuracy
#     worst_acc, group_accs = compute_worst_group_accuracy(dataloader['test'], gnnNets)
#     group_accs_str = ", ".join(f"class {k}: {v:.4f}" for k, v in sorted(group_accs.items()))
#     print(f"Worst-Group Accuracy: {worst_acc:.4f} | Per-Class: [{group_accs_str}]")
#     append_record(f"worst_group_acc: {worst_acc:.4f}, per_class: {group_accs}")

#     return test_state['acc']





def evaluate_GC(eval_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    gnnNets.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            logits, probs, _, _, _, _, _, _ = gnnNets(batch)
            if data_args.dataset_name == 'clintox':
                batch.y = torch.tensor([torch.argmax(i).item() for i in batch.y]).to(model_args.device)
            loss = criterion(logits, batch.y)


            ## record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())

        eval_state = {'loss': np.average(loss_list),
                      'acc': np.concatenate(acc, axis=0).mean()}

    return eval_state




def test_GC(test_dataloader, gnnNets, criterion):
    acc = []
    loss_list = []
    pred_probs = []
    predictions = []
    all_logits = []
    all_labels = []
    gnnNets.eval()

    with torch.no_grad():
        for batch_index, batch in enumerate(test_dataloader):
            logits, probs, active_node_index, _, _, _, _, _ = gnnNets(batch)
            loss = criterion(logits, batch.y)
            
            # # test_subgraph extraction          
            # save_dir = os.path.join('./masking_interpretation_results',
            #                         f"{mcts_args.dataset_name}_"
            #                         f"{model_args.readout}_"
            #                         f"{model_args.model_name}_")
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # plotutils = PlotUtils(dataset_name=data_args.dataset_name)

            # for i, index in enumerate(test_dataloader.dataset.indices[batch_index * train_args.batch_size: (batch_index+1) * train_args.batch_size]):
            #     data = test_dataloader.dataset.dataset[index]
            #     graph = to_networkx(data, to_undirected=True)
            #     if type(active_node_index[i]) == int:
            #         active_node_index[i] = [active_node_index[i]]
            #     print(active_node_index[i])
            #     plotutils.plot(graph, active_node_index[i], x=data.x,
            #                 figname=os.path.join(save_dir, f"example_{i}.png"))
    

            # record
            _, prediction = torch.max(logits, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(batch.y).cpu().numpy())
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
            predictions.append(prediction)
            pred_probs.append(probs)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    auc_score = calculate_auc(all_logits, all_labels)

    test_state = {'loss': np.average(loss_list),
                  'acc': np.average(np.concatenate(acc, axis=0).mean()),
                  'auc': auc_score}

    pred_probs = torch.cat(pred_probs, dim=0).cpu().detach().numpy()
    predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
    return test_state, pred_probs, predictions

def calculate_auc(all_logits, all_labels):
    """
    Calculates AUC-ROC.
    all_logits: torch.Tensor or np.array of shape (N, num_classes)
    all_labels: torch.Tensor or np.array of shape (N)
    """
    # Convert to probabilities via Softmax
    probs = F.softmax(all_logits.clone().detach(), dim=1).numpy()
    labels = all_labels.clone().detach().numpy()
    
    num_classes = probs.shape[1]
    
    if num_classes == 2:
        # Binary case: use probabilities of the positive class
        return roc_auc_score(labels, probs[:, 1])
    else:
        # Multi-class case: OvR (One-vs-Rest) is standard
        return roc_auc_score(labels, probs, multi_class='ovr')
    
def compute_fidelity(test_dataloader, gnnNets):
    """
    High fid+ = removing explanation changes predictions (explanation is necessary/faithful).
    Low fid-  = keeping only explanation preserves predictions (explanation is sufficient).
    """
    complement_matches = []
    subgraph_matches = []

    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = batch.to(model_args.device)

            # 1. Original prediction on full graph
            logits_orig, _, active_node_index, _, _, _, _, _ = gnnNets(batch)
            _, y_pred_orig = torch.max(logits_orig, dim=-1)

            # 2. Build a global node mask from active_node_index
            num_nodes = batch.x.size(0)
            node_mask = torch.zeros(num_nodes, device=model_args.device)

            node_offsets = [0]
            for i in range(batch.batch[-1].item() + 1):
                node_offsets.append(
                    node_offsets[-1] + (batch.batch == i).sum().item()
                )

            for i, nodes in enumerate(active_node_index):
                if isinstance(nodes, int):
                    nodes = [nodes]
                elif not isinstance(nodes, list):
                    nodes = (
                        [nodes] if not hasattr(nodes, '__len__') else list(nodes)
                    )
                for n in nodes:
                    node_mask[node_offsets[i] + n] = 1.0

            # 3. Subgraph-only prediction (for fid-)
            x_sub = batch.x * node_mask.unsqueeze(1)
            data_sub = Batch(
                x=x_sub, edge_index=batch.edge_index, batch=batch.batch
            )
            logits_sub, _, _, _, _, _, _, _ = gnnNets(data_sub)
            _, y_pred_sub = torch.max(logits_sub, dim=-1)

            # 4. Complement-only prediction (for fid+)
            x_comp = batch.x * (1.0 - node_mask).unsqueeze(1)
            data_comp = Batch(
                x=x_comp, edge_index=batch.edge_index, batch=batch.batch
            )
            logits_comp, _, _, _, _, _, _, _ = gnnNets(data_comp)
            _, y_pred_comp = torch.max(logits_comp, dim=-1)

            complement_matches.append(
                (y_pred_comp == y_pred_orig).float().cpu().numpy()
            )
            subgraph_matches.append(
                (y_pred_sub == y_pred_orig).float().cpu().numpy()
            )

    pos_fidelity = 1.0 - np.concatenate(complement_matches).mean()
    neg_fidelity = 1.0 - np.concatenate(subgraph_matches).mean()

    # Characterization score (weighted harmonic mean, GraphFramEx Eq.)
    if pos_fidelity > 0 and neg_fidelity < 1:
        charact = 1.0 / (0.5 / pos_fidelity + 0.5 / (1.0 - neg_fidelity))
    else:
        charact = 0.0

    return pos_fidelity, neg_fidelity, charact


def compute_worst_group_accuracy(test_dataloader, gnnNets):
    """
    Compute worst-group accuracy where each group is a distinct class label.

    Groups g = (y) are defined by the graph label. Per-class accuracy is
    computed, and the minimum across all classes is returned.

    Returns:
        worst_acc   : float, accuracy of the worst-performing class
        group_accs  : dict, {class_id: accuracy} for every class
    """
    all_preds = []
    all_labels = []

    gnnNets.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            logits, _, _, _, _, _, _, _ = gnnNets(batch)
            _, prediction = torch.max(logits, dim=-1)
            all_preds.append(prediction.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    groups = np.unique(all_labels)
    group_accs = {}
    for g in groups:
        mask = all_labels == g
        group_accs[int(g)] = float((all_preds[mask] == all_labels[mask]).mean())

    worst_acc = min(group_accs.values())
    return worst_acc, group_accs


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    # print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }

    pth_name = f"{model_name}_{model_type}_{model_args.readout}_latest.pth"
    best_pth_name = f'{model_name}_{model_type}_{model_args.readout}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        torch.save(gnnNets, os.path.join(ckpt_dir, best_pth_name) )
    gnnNets.to(model_args.device)



if __name__ == '__main__':
    if os.path.isfile("./log/hyper_search.txt"):
        os.remove("./log/hyper_search.txt")

    if model_args.cont:
        model_type = 'cont'
    else:
        model_type = 'var'

    accuracy = train_GC(model_type)