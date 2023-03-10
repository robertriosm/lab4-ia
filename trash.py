
"""

TRAINING_SET_SIZE = 200

x = np.linspace(-10, 30, TRAINING_SET_SIZE)

X = np.vstack(
    (
        np.ones(TRAINING_SET_SIZE),
        x,
        x ** 2,
        x ** 3,
    )
).T

y = (5 + 2 * x ** 3 + np.random.randint(-9000, 9000, TRAINING_SET_SIZE)).reshape(
    TRAINING_SET_SIZE,
    1
)

"""