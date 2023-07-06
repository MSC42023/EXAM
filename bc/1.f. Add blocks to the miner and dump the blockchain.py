import hashlib

class Blockchain:
    def __init__(self):
        self.chain = []
        self.difficulty = 4

    def add_block(self, transactions):
        if not self.chain:
            previous_block_hash = "00000000000000000000000000000000"
        else:
            previous_block = self.chain[-1]
            previous_block_hash = previous_block['block_hash']

        block_hash, nonce = self.mine_block(previous_block_hash, transactions)
        self.chain.append({
            'block_hash': block_hash,
            'previous_block_hash': previous_block_hash,
            'transactions': transactions,
            'nonce': nonce
        })

    def mine_block(self, previous_block_hash, transactions):
        nonce = 0
        prefix = '0' * self.difficulty
        while True:
            block_data = str(nonce) + previous_block_hash + transactions
            block_hash = hashlib.sha256(block_data.encode()).hexdigest()
            if block_hash.startswith(prefix):
                return block_hash, nonce
            nonce += 1

    def dump_blockchain(self):
        for block in self.chain:
            print("Block Hash:", block['block_hash'])
            print("Previous Block Hash:", block['previous_block_hash'])
            print("Transactions:", block['transactions'])
            print("Nonce:", block['nonce'])
            print("-----------------------")

# Create a new blockchain instance
blockchain = Blockchain()

# Add blocks to the blockchain
blockchain.add_block("Transaction data 1")
blockchain.add_block("Transaction data 2")
blockchain.add_block("Transaction data 3")

# Dump the blockchain
blockchain.dump_blockchain()
