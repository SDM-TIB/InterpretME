import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('You should call this file with exactly one parameter, i.e., the path to the file containing the query!')
        exit(1)

    query_file = sys.argv[1]
    try:
        query = open(query_file, 'r', encoding='utf8').read()
        from InterpretME import federated
        federated(query)
    except FileNotFoundError:
        print('No such file:', query_file)
