import binascii import
collectionsimport
datetime
from client import Client from
Crypto.Hash import SHA
from Crypto.Signature import PKCS1_v1_5
class Transaction:
def init (self, sender, recipient, value):self.sender =
sender
self.recipient = recipientself.value = value self.time = datetime.datetime.now()
def to_dict(self):
identity = "Genesis" if self.sender = "Genesis" elseself.sender.identity return collections.OrderedDict(
{
"sender": identity, "recipient": self.recipient,"value": self.value, "time": self.time,
} )
def sign_transaction(self):
private_key = self.sender._private_keysigner = PKCS1_v1_5.new(private_key)
h = SHA.new(str(self.to_dict()).encode("utf8")) return binascii.hexlify(signer.sign(h)).decode("ascii")
Dinesh = Client() Ramesh = Client()
t = Transaction(Dinesh, Ramesh.identity, 5.0) print("\nTransaction Recipient:\n", t.recipient)# print("\nTransaction Sender:\n", t.sender)
