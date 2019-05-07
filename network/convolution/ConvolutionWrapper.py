from network.base.NetworkWrapper import BaseWrapper, BaseWrapper2
import numpy as np

class ConvolutionWrapper(BaseWrapper):
    def get_state(self):
        vision = self.player.take_a_look()
        vision_distance = self.player.vision_distance
        state = [np.zeros((1, 3, vision_distance * 2 + 1, vision_distance * 2 + 1)), np.zeros((2 + self.model.outputs))]
        offset = 0
        square_index = 0
        for y in range(-vision_distance, vision_distance + 1):
            for x in range(-offset, offset + 1):
                state[0].itemset((0, 0, x + vision_distance, y + vision_distance), 1 if vision[square_index].food else 0)
                state[0].itemset((0, 1, x + vision_distance, y + vision_distance), 1 if vision[square_index].trap else 0)
                state[0].itemset((0, 2, x + vision_distance, y + vision_distance), 1 if vision[square_index].stone else 0)
                square_index += 1
            offset += 1 if y < 0 else -1
        state[1].itemset((0), self.player.food)
        state[1].itemset((1), self.player.stones)
        state[1].itemset((int(self.player.last_action) + 2), 1)
        return state

class ConvolutionWrapper2(BaseWrapper2):
    def get_state(self):
        vision = self.player.take_a_look()
        vision_distance = self.player.vision_distance
        state = [np.zeros((1, 2, vision_distance * 2 + 1, vision_distance * 2 + 1)), np.zeros((1 + self.model.outputs))]
        offset = 0
        square_index = 0
        for y in range(-vision_distance, vision_distance + 1):
            for x in range(-offset, offset + 1):
                state[0].itemset((0, 0, x + vision_distance, y + vision_distance), 1 if vision[square_index].food else 0)
                state[0].itemset((0, 1, x + vision_distance, y + vision_distance), 1 if vision[square_index].trap else 0)
                square_index += 1
            offset += 1 if y < 0 else -1
        state[1].itemset((0), self.player.food)
        state[1].itemset((int(self.player.last_action) + 1), 1)
        return state
