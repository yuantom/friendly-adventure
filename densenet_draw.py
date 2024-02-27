import matplotlib.pyplot as plt

method='\outlook_densenet121_data_augment_pretrain'
save_path='D:\python_study\machine_learning_Py/result\densenet'
loss_path=save_path+method+'/pretrained_loss_cor.txt'
with open(loss_path) as file1:
    lines1=file1.readlines()

train_loss=[]
eval_loss=[]
train_cor=[]
eval_cor=[]
for i in range(len(lines1)):
    train_loss.append(float(lines1[i].strip().split()[1]))
    eval_loss.append(float(lines1[i].strip().split()[5]))
    train_cor.append(float(lines1[i].strip().split()[3]))
    eval_cor.append(float(lines1[i].strip().split()[7]))
    print(i)

epoch_list=[]
for j in range(len(lines1)):
    epoch_list.append(j)

plt.plot(epoch_list,train_loss,'r')
plt.plot(epoch_list,eval_loss,'b')
plt.legend(['train loss','eval loss'])
# plt.title('loss')
plt.xlabel('epoches')
plt.ylabel('loss')
plt.savefig(save_path+method+'/loss.jpg')
plt.close()



plt.plot(epoch_list,train_cor,'r')
plt.plot(epoch_list,eval_cor,'b')
plt.legend(['train accuracy','eval accuracy'])
# plt.title('accuracy')
plt.xlabel('epoches')
plt.ylabel('accuracy')
plt.savefig(save_path+method+'/accuracy.jpg')
plt.close()