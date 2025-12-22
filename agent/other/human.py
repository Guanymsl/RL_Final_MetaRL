class ManualAgent:
    def __init__(self):
        self.reward = 0

    def step(self, state):
        action_name = {
            0: "call",
            1: "raise",
            2: "fold",
            3: "check",
        }
        width = 49
        legal_actions_str = ", ".join(
            f"{a}: {action_name[a]}"
            for a in state["legal_actions"].keys()
        )
        last_two_actions = state["action_record"][-2:]
        print(
            "\n" + "=" * width + "\n"
            f"| {'Legal Actions':^{width-4}} |\n"
            f"| {legal_actions_str:^{width-4}} |\n"
            + "-" * width + "\n"
            f"| {'Hand':<8}| {str(state['raw_obs']['hand']):<35} |\n"
            f"| {'Public':<8}| {str(state['raw_obs']['public_cards']):<35} |\n"
            f"| {'History':<8}| {str(last_two_actions):<35} |\n"
            + "=" * width
        )
        while True:
            raw = input("Choose action: ")

            try:
                action = int(raw)
            except ValueError:
                print("Please enter an integer.")
                continue

            if action not in state["legal_actions"]:
                print(f"Illegal action.")
                continue

            break
        return action
