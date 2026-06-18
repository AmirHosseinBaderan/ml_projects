from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

base_dir = Path('./data')
cat_dir = base_dir / 'train/cats/'
dog_dir = base_dir / 'train/dogs/'

train_cats = len(list(cat_dir.glob('*')))
train_dogs = len(list(dog_dir.glob('*')))

print(f"Cats : {train_cats}")
print(f"Dogs : {train_dogs}")

# Create dataset
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir / 'train',
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir / 'validation',
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# check class 
print(train_ds.class_names)
for images,labels in train_ds.take(1):
    print(images.shape)
    print(labels.shape)
    
    # show one batch  
    for i in range(9):
        ax = plt.subplot(3,3,i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        
        plt.title(
            train_ds.class_names[labels[i]]
        )
        
        plt.axis("off")
        
plt.show()