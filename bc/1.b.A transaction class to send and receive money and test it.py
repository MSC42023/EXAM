import hashlib
import time

class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = time.time()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        sha = hashlib.sha256()
        hash_str = str(self.sender) + str(self.receiver) + str(self.amount) + str(self.timestamp)
        sha.update(hash_str.encode('utf-8'))
        return sha.hexdigest()

    def verify_transaction(self):
        if self.hash != self.calculate_hash():
            return False
        return True

# Testing the Transaction class
if __name__ == '__main__':
    # Creating some sample transactions
    transaction1 = Transaction("Alice", "Bob", 10)
    transaction2 = Transaction("Bob", "Charlie", 5)
    transaction3 = Transaction("Charlie", "Alice", 3)

    # Verifying transactions
    print("Transaction 1: ", transaction1.verify_transaction())
    print("Transaction 2: ", transaction2.verify_transaction())
    print("Transaction 3: ", transaction3.verify_transaction())
