def draw_progress (percentage,elapsed_time):
    '''Function to draw a progress bar in the console.'''
    
    bar_length = 40
    filled_length = int(bar_length * percentage // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {percentage:.2f}%  -  elapsed Time: {elapsed_time:.2f}s', end='\r')
    if percentage >= 99:
        print(f'\rProgress: |{'█' * 40}| 100% - elapsed Time: {elapsed_time:.2f}s')

    return