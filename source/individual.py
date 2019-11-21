import abc


class BaseIndividual(abc.ABC):
    @staticmethod
    def get_action(state):
        pass
