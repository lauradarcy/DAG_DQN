import numpy as np


class ExperienceReplay:
    def __init__(self, buffer_size=50000, unusual_sample_factor=0.99):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,features,reward,done,time]
        self.buffer = []
        self.buffer_size = buffer_size

        assert unusual_sample_factor <= 1, "unusual_sample_factor has to be <= 1"
        # Setting this value to a low number over-samples experience that had unusually high or
        # low rewards
        self.unusual_sample_factor = unusual_sample_factor

    def __add__(self, experience):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer.append(experience)
        # Keep the extreme values and most recent values near the end of the buffer to keep them
        self.buffer = sorted(self.buffer, key=lambda replay: (abs(replay[2]),replay[4]))
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size:]


    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        # We want to over-sample frames where things happened. So we'll sort the buffer on the absolute reward
        # (either positive or negative) and apply a geometric probability in order to bias our sampling to the
        # earlier (more extreme) replays
        buffer = sorted(self.buffer, key=lambda replay: (abs(replay[2]),replay[4]), reverse=True)
        p = np.array([self.unusual_sample_factor ** i for i in range(len(buffer))])
        p = p / sum(p)
        sample_idxs = np.random.choice(np.arange(len(buffer)), size=size, p=p)
        sample_output = [buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output







