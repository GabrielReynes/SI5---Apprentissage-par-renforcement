from collections import deque
import random


class AgentMemory:
    """
    This class is used to store an agent's game memory.
    This buffer is of a fixed size, and deletes automatically its first element when its maximum size is met.
    """

    def __init__(self, buffer_length):
        self.buffer = deque(maxlen=buffer_length)

    def push(self, *args):
        """
        Add all the given argument as a tuple inside the memory buffer.
        If the buffer is already at its maximum size, it automatically deletes its first element.
        """
        self.buffer.append(args)

    def sample(self, size):
        """
        Returns a new list containing a defined number of random elements contained inside the AgentMemory instance.
        :param size: The size of the sample.
        :return: A random sample of the AgentMemory instance
        """
        cropped_size = min(len(self.buffer), size)
        return random.sample(self.buffer, cropped_size)

    def clear(self):
        """
        Clears the AgentMemory instance buffer. Deleting all of its previously stored elements.
        :return:
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
