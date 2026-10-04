"""Cryptographic fingerprinting (Level 1). Answers: is this the exact
artifact we expected? Says nothing about behavior."""
import hashlib

CHUNK_SIZE = 1024 * 1024


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
