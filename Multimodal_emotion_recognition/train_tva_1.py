import torch
from torch import nn
import models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_utils import save_model, load_model, weighted_accuracy, unweighted_accuracy, weighted_precision, unweighted_precision
import sys
import pandas as pd

def initiate(hyp_params, train_loader, dev_loader, test_loader):
    tva_model = getattr(models, 'TVAModel_Cross')(hyp_params)
    tva_model = tva_model.double().to('cuda')
    #import pdb
    #pdb.set_trace()
    optimizer = getattr(optim, hyp_params.optim)(filter(lambda p: p.requires_grad, tva_model.parameters()), lr=hyp_params.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True) # 优化学习率
    settings = {'tva_model': tva_model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,
                }
    return train_model(settings, hyp_params, train_loader, dev_loader, test_loader)



def train_model(settings, hyp_params, train_loader, dev_loader, test_loader):
    tva_model = settings['tva_model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    def train(tva_model_, criterion_, optimizer_):
        epoch_loss_total = 0
        tva_model_.train()
        results_=[]
        truths_=[]
        for i_batch, batch_X in enumerate(train_loader):

            x_text, x_vid, x_aud, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3]

            tva_model_.zero_grad()
            batch_size = x_text.size(0)
            preds, _= tva_model_(x_text.double(), x_vid.double(), x_aud.double())
            raw_loss = criterion_(preds, labels)
            raw_loss.backward()

            optimizer_.step()
            epoch_loss_total += raw_loss.item() * batch_size
            results_.append(preds)
            truths_.append(labels)
        results_ = torch.cat(results_)
        truths_ = torch.cat(truths_)
        return epoch_loss_total / hyp_params.n_train, results_, truths_

    def evaluate(tva_model_, criterion_, test=False):
        tva_model_.eval()
        loader = test_loader if test else dev_loader
        total_loss = 0.0
        results_ = []
        truths_ = []
        ints_ = [] # intermediate embeddings
        with torch.no_grad():
            for i_batch, batch_X in enumerate(loader):

                x_text, x_vid, x_aud, labels = batch_X[0], batch_X[1], batch_X[2], batch_X[3]

                batch_size = x_text.size(0)
                preds, x_int = tva_model_(x_text.double(), x_vid.double(), x_aud.double())
                total_loss += criterion_(preds, labels).item() * batch_size
                # Collect the results into dictionary
                results_.append(preds)
                truths_.append(labels)
                ints_.append(x_int)
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_dev)

        results_ = torch.cat(results_)
        truths_ = torch.cat(truths_)
        ints_ = torch.cat(ints_)
        return avg_loss, results_, truths_, ints_

    def perf_eval(results_, truths_):
        results_ = torch.argmax(results_, dim=1)
        truths_ = truths_.cpu().numpy()
        results_ = results_.cpu().numpy()
        results_ = results_.tolist()
        truths_ = truths_.tolist()
        wa_ = weighted_accuracy(truths_, results_)
        uwa_ = unweighted_accuracy(truths_, results_)
        wp_ = weighted_precision(truths_, results_)
        uwp_ = unweighted_precision(truths_, results_)
        return wa_, uwa_, wp_, uwp_

    def make_cm(matrix, col):

        confusion_matrix = pd.DataFrame(matrix,columns=col,index=col)
        return confusion_matrix

    train_loss_list=[]
    train_wa_list=[]
    train_ua_list=[]

    test_loss_list=[]
    test_wa_list=[]
    test_ua_list=[]

    best_val_uwa = 0
    es = 0
    #"""
    for epoch in range(1, hyp_params.num_epochs + 1):
        train_total_loss, train_results, train_truth = train(tva_model, criterion, optimizer)
        train_wa, train_uwa, train_wp, train_uwp=perf_eval(train_results, train_truth)
        val_loss, val_res, val_tru, _ = evaluate(tva_model, criterion, test=False)
        val_wa, val_uwa, val_wp, val_uwp = perf_eval(val_res, val_tru)
        test_loss, tst_res, tst_tru, _ = evaluate(tva_model, criterion, test=True)
        tst_wa, tst_uwa, tst_wp, tst_uwp = perf_eval(tst_res, tst_tru)
        scheduler.step(val_loss)  # Decay learning rate by validation loss
        print("-" * 50)
        print('Epoch {:2d} | Train Total Loss {:5.4f}'.format(epoch, train_total_loss))
        print('Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(val_loss, test_loss))
        print('Valid WA {:5.4f} | UWA {:5.4f} | WP {:5.4f} | UWP {:5.4f}'.format(val_wa, val_uwa, val_wp, val_uwp))
        print('Test WA {:5.4f} | UWA {:5.4f} | WP {:5.4f} | UWP {:5.4f}'.format(tst_wa, tst_uwa, tst_wp, tst_uwp))
        print("-" * 50)
        sys.stdout.flush()

        if val_uwa > best_val_uwa:
            print("Saved model at epoch: ", epoch)
            save_model(tva_model, name='pre_trained_models/final_exp.pth')

            best_val_uwa = val_uwa


        train_loss_list.append(train_total_loss)
        train_wa_list.append(train_wa)
        train_ua_list.append(train_uwa)
        test_loss_list.append(test_loss)
        test_ua_list.append(tst_wa)
        test_wa_list.append(tst_uwa)
    #"""
    model = load_model(name='pre_trained_models/final_exp.pth')
    total = sum([param.nelement() for param in model.parameters()])

    print(total)

    _, results, truths, ints = evaluate(model, criterion, test=True)
    results = torch.argmax(results, dim=1)
    truths = truths.cpu().numpy()
    results = results.cpu().numpy()
    # ints = ints.cpu().numpy()
    from sklearn.metrics import classification_report as cr
    print(cr(truths, results))
    from sklearn.metrics import confusion_matrix as cm
    matrix=cm(truths, results,labels=[0,1,2,3,4,5,6])
    confusion=make_cm(matrix,['ang', 'exc', 'hap', 'sad', 'fru', 'neu', 'sur'])
    print(confusion)

    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np

    confusion_labels=['ang', 'exc', 'hap', 'sad', 'fru', 'neu', 'sur']
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=confusion_labels)

    disp.plot(cmap=plt.cm.Blues)
    plt.savefig('./confusion.jpg')
    plt.show()


    x = list(range(1, len(train_loss_list)+1))
    plt.figure()
    plt.title("train/test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x, train_loss_list, label="train loss")
    plt.plot(x, test_loss_list, label='test loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('./loss.jpg')
    plt.show()

    y = list(range(1, len(train_ua_list)+1))
    fig=plt.figure()
    p1, p2 = fig.subplots(2, 1, sharex=True, sharey=False)
    p1.set_title('train/test accuracy')
    p1.set_ylabel('Weighted accuracy')  # 设置坐标轴名字
    p2.set_ylabel('Unweighted accuracy')
    p2.set_xlabel('epochs')


    p1.plot(y, train_wa_list, label="train weighted accuracy")
    p1.plot(y, test_wa_list, label='test weighted accuracy')
    p1.legend(loc='best')
    p1.grid(True)

    z=list(range(1, len(train_wa_list)+1))
    p2.plot(z, train_ua_list, label="train unweighted accuracy")
    p2.plot(z, test_ua_list, label='test unweighted accuracy')
    p2.legend(loc='best')
    p2.grid(True)
    fig.savefig('./accuracy.jpg')
    fig.show()

    results = results.tolist()
    truths = truths.tolist()
    wa = weighted_accuracy(truths, results)
    uwa = unweighted_accuracy(truths, results)
    wp = weighted_precision(truths, results)
    uwp = unweighted_precision(truths, results)
    print("weighted accuracy:", wa)
    print("unweighted accuracy:", uwa)
    print("weighted precision:", wp)
    print("unweighted precision:", uwp)

    sys.stdout.flush()
