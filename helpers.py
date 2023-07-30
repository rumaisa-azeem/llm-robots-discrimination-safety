import os

def gen_filename(type:str=''):
    output_dir = 'outputs'
    next_filename = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = os.listdir(output_dir)
    if not filenames:
        next_filename = os.path.join(output_dir, f'{type}1.txt')
    else:
        # Find the highest number used in the filenames
        max_number = max([int(os.path.splitext(filename)[0][-1]) for filename in filenames])
        next_number = max_number + 1
        next_filename = os.path.join(output_dir, f'{type}{next_number}.txt')

    return next_filename
