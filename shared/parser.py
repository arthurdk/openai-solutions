import argparse


def create_parser(game):
    """
    Create a generic parser
    :param game: game name
    :return: parser
    """
    parser = argparse.ArgumentParser(description='Learn an AI to solve {} game.'.format(game),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-S', '--stats', action='store_true',
                        help='Display some statistics at the end of the session')
    parser.add_argument('-P', '--publish', action='store_true',
                        help='Publish performance on Gym, require to have a variable API_KEY="YOUR8KEY" in shared/api.py')
    return parser
