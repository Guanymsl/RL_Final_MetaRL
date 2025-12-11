import torch

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return self.__str__()

class CardTensorFormatter:
    def __init__(self):
        self.suit_to_index = {'Hearts': 0, 'Diamonds': 1, 'Clubs': 2, 'Spades': 3}
        self.rank_to_index = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
            '10': 8, 'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
        }

    def cardToMatrix(self, card):
        matrix = torch.zeros((4, 13), dtype=torch.float32)
        suit_idx = self.suit_to_index[card.suit]
        rank_idx = self.rank_to_index[card.rank]
        matrix[suit_idx, rank_idx] = 1
        return matrix

    def cardToTensor(self, hole_cards, community_cards):
        tensor = torch.zeros((6, 4, 13), dtype=torch.float32)

        # Channel 0: Agent's 2 hole cards
        for card in hole_cards:
            tensor[0] += self.cardToMatrix(card)

        # Channel 1: Flop cards
        for card in community_cards["Flop"]:
            tensor[1] += self.cardToMatrix(card)

        # Channel 2: Turn card
        tensor[2] += self.card_to_matrix(community_cards["Turn"][0])

        # Channel 3: River card
        tensor[3] += self.card_to_matrix(community_cards["River"][0])

        # Channel 4: All public cards (Flop + Turn + River)
        for card in community_cards["Flop"] + community_cards["Turn"] + community_cards["River"]:
            tensor[4] += self.card_to_matrix(card)

        # Channel 5: All hole and public cards (Agent's hole cards + all community cards)
        for card in hole_cards + community_cards["Flop"] + community_cards["Turn"] + community_cards["River"]:
            tensor[5] += self.card_to_matrix(card)

        return tensor
