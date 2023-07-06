import hashlib
import time

def calculate_hash(timestamp, data, previous_hash):
    sha = hashlib.sha256()
    sha.update((str(timestamp) + str(data) + str(previous_hash)).encode('utf-8'))
    return sha.hexdigest()

def create_genesis_block():
    timestamp = time.time()
    data = "Genesis Block"
    previous_hash = "0"
    hash = calculate_hash(timestamp, data, previous_hash)
    return {'timestamp': timestamp, 'data': data, 'previous_hash': previous_hash, 'hash': hash}

def add_block(chain, data):
    previous_block = chain[-1]
    timestamp = time.time()
    previous_hash = previous_block['hash']
    hash = calculate_hash(timestamp, data, previous_hash)
    block = {'timestamp': timestamp, 'data': data, 'previous_hash': previous_hash, 'hash': hash}
    chain.append(block)

# Testing the Blockchain
if __name__ == '__main__':
    blockchain = [create_genesis_block()]

    add_block(blockchain, "Block 1 Data")
    add_block(blockchain, "Block 2 Data")

    is_chain_valid = all(blockchain[i]['hash'] == calculate_hash(blockchain[i]['timestamp'], blockchain[i]['data'], blockchain[i]['previous_hash'])
                         and blockchain[i]['previous_hash'] == blockchain[i-1]['hash']
                         for i in range(1, len(blockchain)))

    print("Is blockchain valid? ", is_chain_valid)
