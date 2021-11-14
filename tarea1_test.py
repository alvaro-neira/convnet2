import sys
import statistics
from tarea1 import mapping
from tarea1.mapping import animals

if __name__ == '__main__':
    for (i, item) in enumerate(animals):
        print(f"{i}->'{item}'")


def get_animal(path_str):
    if not path_str:
        sys.exit('empty string')
    if "/" not in path_str:
        sys.exit('invalid string')
    count = path_str.count("/")
    if count < 2:
        sys.exit('invalid string 2')
    splitted = path_str.split("/")
    return splitted[len(splitted) - 2].strip().lower()


print(get_animal('data/test_images/bee/026_00119802.jpg'))
print(get_animal('/content/convnet2/data/test_images/bee/026_00119695.jpg'))
print(get_animal('/content/convnet2/data/test_images/cow/081_00122082.jpg'))
print(get_animal('/content/convnet2/data/test_images/sheep/258_00124921.jpg'))
print(get_animal('/content/convnet2/data/test_images/cat/064_00122151.jpg'))
print(get_animal('/content/convnet2/data/test_images/crocodile/084_00126855.jpg'))
print(get_animal('a//'))
print(get_animal('/content/convnet2/data/test_images/crab/082_00125929.jpg'))
print(get_animal('/content/convnet2/data/test_images/elephant/103_00125872.jpg'))

arr=['/content/convnet2/data/test_images/cat/064_00122151.jpg', 'data/test_images/crab/082_00125736.jpg', 'data/test_images/cat/064_00122174.jpg', 'data/test_images/cat/064_00122190.jpg', 'data/test_images/bee/026_00119842.jpg', 'data/test_images/cat/064_00122063.jpg', 'data/test_images/crab/082_00125813.jpg', 'data/test_images/cat/064_00122032.jpg', 'data/test_images/cat/064_00122140.jpg', 'data/test_images/cat/064_00122110.jpg']

# print(map(get_animal, arr[:5]))

ap_dict = {'total':[]}
p1_dict = {'total':[]}
for (i, animal) in enumerate(animals):
    ap_dict[animal]=[]
    p1_dict[animal]=[]

ap_dict['cat'].append(69.0)
ap_dict['cat'].append(10.0)
ap_dict['cat'].append(11.0)
print(ap_dict)
print(p1_dict)

print(f"mean={statistics.mean(ap_dict['cat'])}")
