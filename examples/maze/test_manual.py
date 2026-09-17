from game import Game, EnvironmentConfig

if __name__ == "__main__":
    env = Game(config=EnvironmentConfig(params={
        "agent": {"discrete_actions": True}
    }))
    env.run(total=10)