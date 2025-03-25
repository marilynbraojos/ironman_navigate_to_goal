import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/michaelangelo/ironman_navigate_to_goal/install/ironman_chase_object'
