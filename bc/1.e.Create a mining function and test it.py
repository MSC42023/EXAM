import hashlib

def mine_block(previous_block_hash, transactions, difficulty):
    nonce = 0
    prefix = '0' * difficulty
    while True:
        block_data = str(nonce) + previous_block_hash + transactions
        block_hash = hashlib.sha256(block_data.encode()).hexdigest()
        if block_hash.startswith(prefix):
            return block_hash, nonce
        nonce += 1
previous_block_hash = "00000000000000000000000000000000"
transactions = "Transaction data"
difficulty = 4

block_hash, nonce = mine_block(previous_block_hash, transactions, difficulty)
print(f"Block hash: {block_hash}")
print(f"Nonce: {nonce}")
