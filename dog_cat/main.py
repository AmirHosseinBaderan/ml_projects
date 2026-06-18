from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path('./data')
cat_dir = base_dir / 'train/cats/'
dog_dir = base_dir / 'train/dogs/'

train_cats = len(list(cat_dir.glob('*')))
train_dogs = len(list(dog_dir.glob('*')))

print(f"Cats : {train_cats}")
print(f"Dogs : {train_dogs}")


images = list(cat_dir.glob("*.jpg"))
plt.figure(figsize=(10,0))

for i in range(9):
    plt.subplot(3,3,i+1)
    
    img = plt.imread(images[i])
    plt.imshow(img)
    plt.axis("off")
    
plt.show()
    