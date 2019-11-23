import abc


class BaseIndividual(abc.ABC):
    @staticmethod
    def get_action(state, avail_actions):
        pass
