# Revival: Convert Minecraft to Real World
# CS 175: Project in Artificial Intelligence (Spring 2019)
#
# revival.py
#

from cyclegan import CycleGAN
from dataloader import RevivalDataLoader

# import warnings
# warnings.filterwarnings("ignore")

EPOCH = 100
LOG_PER_ITERATION = 200
TRAIN = True
LOAD = False

def main():
    train_dloader = RevivalDataLoader('./datasets/trainA', './datasets/trainB')
    test_dloader = RevivalDataLoader('./datasets/testA', './datasets/testB', is_train=False)
    cyclegan = CycleGAN()
    if LOAD:
        cyclegan.load_model()
    cyclegan.create_optimizer()
    if TRAIN:
        for epoch in range(1, EPOCH + 1):
            for iteration, data in enumerate(train_dloader, 1):
                cyclegan.train(data)
                if iteration % LOG_PER_ITERATION == 0:
                    cyclegan.print_train_log(epoch, iteration)

            cyclegan.save_train_log('./train_log.csv')
            cyclegan.save_model()
    else:
        for index, data in enumerate(test_dloader, 1):
            cyclegan.test(data, index)


if __name__ == '__main__':
    main()
