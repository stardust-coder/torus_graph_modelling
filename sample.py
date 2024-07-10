from multiprocess import Pool, Manager
import functools

def myFunc(items,items2, a, b):
    data = (a,b)
    items.append(data)
    items2.append(a)


if __name__ == '__main__':
    manager = Manager()
    items = manager.list()
    items2 = manager.list()

    with Pool(2) as p:
        p.starmap(functools.partial(myFunc, items,items2),[(0,1),(2,3)])

    print(items)
    print(items2)