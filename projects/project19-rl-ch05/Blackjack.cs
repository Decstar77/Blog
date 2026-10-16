using System.Diagnostics;

namespace rl
{
    enum GameResult
    {
        Loss = -1,
        Draw = 0,
        Win = 1,
        Playing = 2,
    }

    class BlackJack
    {
        // 0 = Ace, 10 = Face card or 10
        public List<int> PlayerCards = new List<int>();
        public List<int> DealerCards = new List<int>();
        private Random random = new Random();

        // DealerCards[0] is the showing card, DealerCards[1] is face down.
        public GameResult NewGame()
        {
            PlayerCards.Clear();
            DealerCards.Clear();
            PlayerCards.Add(DrawCard());
            PlayerCards.Add(DrawCard());
            DealerCards.Add(DrawCard());
            DealerCards.Add(DrawCard());

            if (IsNatural(PlayerCards))
            {
                return IsNatural(DealerCards) ? GameResult.Draw : GameResult.Win;
            }
            return GameResult.Playing;
        }

        public GameResult RandomGame()
        {
            PlayerCards.Clear();
            DealerCards.Clear();

            int sum = random.Next(12, 22);
            bool usableAce = random.Next(2) == 1;
            int showing = random.Next(1, 11);

            if (usableAce)
            {
                // Ace counts 11, so the other card is sum - 11 (1..10; 1 means a second ace)
                PlayerCards.Add(0);
                int other = sum - 11;
                PlayerCards.Add(other == 1 ? 0 : other);
            }
            else
            {
                // Hard total with no aces, e.g. 10 + 2 .. 10 + 10; 21 needs three cards
                if (sum <= 20) { PlayerCards.Add(10); PlayerCards.Add(sum - 10); }
                else { PlayerCards.Add(10); PlayerCards.Add(9); PlayerCards.Add(2); }
            }

            DealerCards.Add(showing == 1 ? 0 : showing);
            DealerCards.Add(DrawCard()); // hidden card

            return GameResult.Playing;
        }

        public int PlayerSum => GetHandValue(PlayerCards);

        // 1 = Ace, 2..10
        public int DealerShowing => DealerCards[0] == 0 ? 1 : DealerCards[0];

        public bool PlayerHasUsableAce => HasUsableAce(PlayerCards);

        public bool IsNatural(List<int> cards)
        {
            return cards.Count == 2 && GetHandValue(cards) == 21;
        }

        public bool HasUsableAce(List<int> cards)
        {
            int hardSum = cards.Sum(c => c == 0 ? 1 : c);
            return cards.Contains(0) && hardSum + 10 <= 21;
        }

        public int GetHandValue(List<int> cards)
        {
            int sum = cards.Sum(c => c == 0 ? 1 : c);
            int aceCount = cards.Count(c => c == 0);
            while (aceCount > 0 && sum + 10 <= 21)
            {
                sum += 10;
                aceCount--;
            }
            return sum;
        }

        public GameResult Play(int action) // 0 = stand, 1 = hit
        {
            if (action == 1)
            {
                PlayerCards.Add(DrawCard());
                int hand = GetHandValue(PlayerCards);
                return hand > 21 ? GameResult.Loss : GameResult.Playing;
            }
            else
            {
                while (GetHandValue(DealerCards) < 17)
                {
                    DealerCards.Add(DrawCard());
                }

                int playerHand = GetHandValue(PlayerCards);
                int dealerHand = GetHandValue(DealerCards);
                if (dealerHand > 21)
                {
                    return GameResult.Win;
                }

                if (playerHand > dealerHand)
                {
                    return GameResult.Win;
                }
                else if (playerHand == dealerHand)
                {
                    return GameResult.Draw;
                }
                else if (playerHand < dealerHand)
                {
                    return GameResult.Loss;
                }

                throw new UnreachableException($"Game should be over");
            }
        }

        private int DrawCard()
        {
            // 13 ranks: Ace, 2-9, 10, J, Q, K (infinite deck)
            int rank = random.Next(13);
            return rank == 0 ? 0 : Math.Min(rank + 1, 10);
        }
    }
}