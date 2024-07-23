import os
import pickle
import shutil

c=1
flit=[]
with open('indoor_test.pkl', 'rb') as f:
    cap = pickle.load(f)

    for k, v in cap.items():
        print(k)
        name=k
        flit.append(name)
        if os.path.isfile(os.path.join('/mnt/e/Datasets/indoor_train',name)):
            shutil.move(os.path.join('/mnt/e/Datasets/indoor_train',name),os.path.join('/mnt/e/Datasets/indoor_val_test',name))

        #     print(k)
        # c+=1
print(len(flit))
# with open('test_augment.pkl', 'rb') as f:
#     cap = pickle.load(f)
#     for k, v in cap.items():
#         name=str(k).split('/')[-1]
#         if not name in flit:
#             print(name)