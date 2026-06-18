from pathlib import Path

base_dir = Path('./data').parent

train_cats = len(list((base_dir / '/train/cats/').glob('*')))
train_dogs = len(list((base_dir / '/train/dogs/').glob('*')))

print(f"Cats : {train_cats}")
print(f"Dogs : {train_dogs}")