from datasets import load_dataset

ds = load_dataset('mohanty/PlantVillage', 'default')
print(ds)
print(list(ds.keys()))
for split_name in ds:
    split = ds[split_name]
    print('\nSPLIT:', split_name)
    print('columns:', split.column_names)
    print('features:', split.features)
    print('first row keys:', list(split[0].keys()))
    break
