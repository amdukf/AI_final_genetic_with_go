package main

import (
	"fmt"
	"math/rand"
)

// Chromosome represents a chromosome in the genetic algorithm.
type Chromosome struct {
	gens             []int
	fitness          float32
	fitnessRatio     float32
	probabilityRange [2]float32
	summation        float32
}

// Genetic represents the genetic algorithm.
type Genetic struct {
	chromosomeLength int
	populationSize   uint32
	perMutation      float32
	population       []*Chromosome
	maxiter          uint32
	crossoverType    CrossoverType
	summation        float32
}

// CrossoverType represents the type of crossover used in genetic algorithm.
type CrossoverType int

const (
	OnePoint CrossoverType = iota
	TwoPoint
	Uniform
)

// NewChromosome initializes a new Chromosome instance.
func NewChromosome(gens []int) *Chromosome {
	return &Chromosome{
		gens:             gens,
		fitness:          0.0,
		fitnessRatio:     0.0,
		probabilityRange: [2]float32{0.0, 0.0},
	}
}

// Intersects calculates the number of intersecting elements in the chromosome.
func (c *Chromosome) Intersects() float32 {
	fitness := 0.0
	for i := 0; i < len(c.gens)-1; i++ {
		for j := i + 1; j < len(c.gens); j++ {
			if c.gens[i] == c.gens[j] {
				fitness += 1.0
			}
		}
	}
	return float32(fitness)
}

// Fitness calculates and returns the fitness of the chromosome.
func (c *Chromosome) Fitness() float32 {
	if c.fitness != 0.0 {
		return c.fitness
	}

	c.fitness = 1.0 / (c.Intersects() + c.Epsilon())
	return c.fitness
}

// FitnessRatio calculates and returns the fitness ratio of the chromosome.
func (c *Chromosome) FitnessRatio() float32 {

	return c.Fitness() / c.summation
}

// SetProbabilityRange sets the probability range for chromosome selection.
func (c *Chromosome) SetProbabilityRange(rangeValue [2]float32) {
	c.probabilityRange = rangeValue
}

// IsChosen checks if the chromosome is chosen based on the given number.
func (c *Chromosome) IsChosen(number float32) bool {
	if c.probabilityRange == [2]float32{0.0, 0.0} {
		panic("probability range must be valid or not None in Chromosome structure!")
	}

	return number >= c.probabilityRange[0] && number < c.probabilityRange[1]
}

// Epsilon returns a small value.
func (c *Chromosome) Epsilon() float32 {
	return 0.00000001
}

// Print prints the chromosome details.
func (c *Chromosome) Print() {
	fmt.Printf("gens: %v, intersects: %v, fitness: %v\n", c.gens, c.Intersects(), c.Fitness())
}

// NewGenetic initializes a new Genetic instance.
func NewGenetic(chromosomeLength int, populationSize uint32, perMutation float32, maxiter uint32, crossoverType CrossoverType) *Genetic {
	if perMutation > 1.0 {
		panic("per_mutation argument could not be greater than 1 !")
	}

	genetic := &Genetic{
		chromosomeLength: chromosomeLength,
		populationSize:   populationSize,
		perMutation:      perMutation,
		population:       make([]*Chromosome, 0),
		maxiter:          maxiter,
		crossoverType:    crossoverType,
	}
	genetic.initPopulation()
	return genetic
}

// InitPopulation initializes the initial population.
func (g *Genetic) initPopulation() {
	// Initialize population with some predefined data for testing
	dataset := [][]int{
		{5, 5, 15, 15, 2, 2, 7, 7, 17, 17},
		{1, 15, 15, 25, 25, 25, 12, 17, 22, 3},
		{10, 10, 10, 20, 20, 7, 2, 12, 12, 3},
		{1, 10, 15, 15, 2, 2, 7, 7, 12, 22},
		{5, 5, 20, 25, 20, 25, 12, 12, 17, 3},
		{10, 10, 20, 15, 25, 25, 2, 7, 22, 22},
		{5, 5, 15, 20, 20, 7, 7, 7, 12, 22},
		{10, 15, 15, 25, 20, 2, 2, 12, 17, 17},
		{10, 10, 20, 25, 25, 2, 7, 12, 12, 17},
		{5, 15, 20, 20, 2, 2, 2, 22, 22, 17},
	}

	for _, item := range dataset {
		g.population = append(g.population, NewChromosome(item))
	}
}

// RandomChromosome generates a random chromosome.
func (g *Genetic) randomChromosome() *Chromosome {
	randVec := make([]int, g.chromosomeLength)
	for i := 0; i < g.chromosomeLength; i++ {
		randVec[i] = rand.Intn(26)
	}

	return NewChromosome(randVec)
}

// FitnessSummation calculates the sum of fitness values in the population.
func (g *Genetic) fitnessSummation() float32 {
	var sum float32
	// calculating summation of all fitnesses for all population chromsomes
	for _, chromosome := range g.population {
		sum += chromosome.Fitness()
	}

	for _, chromosome := range g.population {
		chromosome.summation = sum
	}

	return sum
}

// InitProbabilityRange initializes the probability range for each chromosome.
func (g *Genetic) initProbabilityRange() {
	temp := float32(0.0)
	for _, chromo := range g.population {
		temp_ := temp + chromo.FitnessRatio()
		chromo.SetProbabilityRange([2]float32{temp, temp_})
		temp = temp_
	}
}

// ParentSelection selects parents based on fitness and probability range.
func (g *Genetic) parentSelection() []*Chromosome {
	g.summation = g.fitnessSummation()
	g.initProbabilityRange() // Move this line here

	newParents := make([]*Chromosome, 0)
	for len(newParents) < int(g.populationSize) {
		for _, chromo := range g.population {
			randNo := rand.Float32()
			if chromo.IsChosen(randNo) {
				newParents = append(newParents, chromo)
				break
			}
		}
	}

	return newParents
}

// OnePointCrossover performs one-point crossover on two parent chromosomes.
func (g *Genetic) onePointCrossover(parent1, parent2 *Chromosome) (*Chromosome, *Chromosome) {
	// increase random chance of crossover , (breaking the normal plot of statics)
	crossoverPoint := rand.Intn(300) % g.chromosomeLength

	// fmt.Printf("parent 1 gens (one point crossover) : %v\n", parent1.gens)
	// fmt.Printf("parent 2 gens (one point crossover) : %v\n", parent2.gens)
	newChild1 := make([]int, g.chromosomeLength)
	newChild2 := make([]int, g.chromosomeLength)

	copy(newChild1, parent1.gens[:crossoverPoint])
	copy(newChild1[crossoverPoint:], parent2.gens[crossoverPoint:])

	copy(newChild2, parent2.gens[:crossoverPoint])
	copy(newChild2[crossoverPoint:], parent1.gens[crossoverPoint:])

	// fmt.Printf("child 1 gens (one point crossover) : %v\n", newChild1)
	// fmt.Printf("child 2 gens (one point crossover) : %v\n", newChild2)

	return NewChromosome(newChild1), NewChromosome(newChild2)
}

// TwoPointCrossover performs two-point crossover on two parent chromosomes.
func (g *Genetic) twoPointCrossover(parent1, parent2 *Chromosome) (*Chromosome, *Chromosome) {
	ind1 := rand.Intn(g.chromosomeLength)
	ind2 := rand.Intn(g.chromosomeLength-ind1) + ind1

	if ind2 < ind1 {
		ind1, ind2 = ind2, ind1
	}

	newChild1 := make([]int, g.chromosomeLength)
	newChild2 := make([]int, g.chromosomeLength)

	for i := ind1; i <= ind2; i++ {
		newChild1[i] = parent1.gens[i]
		newChild2[i] = parent2.gens[i]
	}

	return NewChromosome(newChild1), NewChromosome(newChild2)
}

// UniformCrossover performs uniform crossover on two parent chromosomes.
func (g *Genetic) uniformCrossover(parent1, parent2 *Chromosome) (*Chromosome, *Chromosome) {
	length := len(parent1.gens)
	child1 := make([]int, length)
	child2 := make([]int, length)

	for i := 0; i < length; i++ {
		randBit := rand.Float32() < 0.5
		if randBit {
			child1[i] = parent1.gens[i]
			child2[i] = parent2.gens[i]
		} else {
			child1[i] = parent2.gens[i]
			child2[i] = parent1.gens[i]
		}
	}

	return NewChromosome(child1), NewChromosome(child2)
}

// Crossover performs crossover based on the specified type.
func (g *Genetic) crossover(parent1, parent2 *Chromosome) (*Chromosome, *Chromosome) {
	switch g.crossoverType {
	case OnePoint:
		return g.onePointCrossover(parent1, parent2)
	case TwoPoint:
		return g.twoPointCrossover(parent1, parent2)
	case Uniform:
		return g.uniformCrossover(parent1, parent2)
	default:
		return g.twoPointCrossover(parent1, parent2)
	}
}

// Recombination performs recombination (crossover) on a list of parent chromosomes.
func (g *Genetic) recombination(parents []*Chromosome) []*Chromosome {
	offsprings := make([]*Chromosome, 0)
	for i := 0; i < len(parents)-1; i += 2 {

		// fmt.Printf("parent 1 gens (recombination) : %v\n", parents[i].gens)
		// fmt.Printf("parent 2 gens (recombination) : %v\n", parents[i + 1].gens)
		child1, child2 := g.crossover(parents[i], parents[i+1])

		// fmt.Printf("child 1 gens (recombination) : %v\n", child1.gens)
		// fmt.Printf("child 2 gens (recombination) : %v\n", child2.gens)

		offsprings = append(offsprings, child1, child2)
	}
	return offsprings
}

// SwapMutation performs swap mutation on a chromosome.
func (g *Genetic) swapMutation(chromosome *Chromosome) *Chromosome {
	newGens := make([]int, len(chromosome.gens))
	copy(newGens, chromosome.gens)

	if rand.Float32() <= g.perMutation {

		// fmt.Printf("gens before sw : %v\n", newGens)
		i := rand.Intn(len(newGens))
		j := rand.Intn(len(newGens))
		newGens[i], newGens[j] = newGens[j], newGens[i]
		// fmt.Printf("i : %v\n", i)
		// fmt.Printf("j : %v\n", j)
		// fmt.Printf("gens after sw : %v\n", newGens)
	}

	return NewChromosome(newGens)
}

// Mutation applies mutation on a list of chromosomes.
func (g *Genetic) mutation(offsprings []*Chromosome) []*Chromosome {
	for i := 0; i < len(offsprings); i++ {
		offsprings[i] = g.swapMutation(offsprings[i])
	}
	return offsprings
}

// MaximumFitness finds the chromosome with the maximum fitness in the population.
func (g *Genetic) maximumFitness(population []*Chromosome) (int, float32) {
	maxI := 0
	maxFit := population[0].Fitness()
	for i := 0; i < len(population); i++ {
		if population[i].Fitness() > maxFit {
			maxFit = population[i].Fitness()
			maxI = i
		}
	}
	return maxI, maxFit
}

// StartLoop runs the genetic algorithm loop.
func (g *Genetic) startLoop() (*Chromosome, []float32) {
	if g == nil {
		panic("Genetic instance is nil.")
	}

	bestFitnesses := make([]float32, 0)
	best := g.randomChromosome()

	for i := uint32(1); i <= g.maxiter; i++ {
		parents := g.parentSelection()

		// for _, chr := range parents {
		// 	fmt.Printf("Current selected parents : %v\n", chr.gens)

		// }

		offsprings := g.recombination(parents)

		// for _, chr := range offsprings {
		// 	fmt.Printf("first offsprings (crossover) : %v\n", chr.gens)

		// }

		offsprings = g.mutation(offsprings)

		// for _, chr := range offsprings {
		// 	fmt.Printf("first offsprings (crossover) : %v\n", chr.gens)

		// }

		g.population = offsprings

		// // Truncate the population to the original size
		// g.population = g.population[:int(g.populationSize)]

		// Update best chromosome
		bestIndex, bestFitness := g.maximumFitness(g.population)
		if bestFitness > best.Fitness() {
			best = g.population[bestIndex]
		}

		// Store best fitness for analysis
		bestFitnesses = append(bestFitnesses, bestFitness)

		// fmt.Printf("Iteration %d, Best Fitness: %v\n", i, bestFitness)
	}

	return best, bestFitnesses
}

func main() {
	chromosomeLength := 10
	populationSize := 10
	perMutation := 0.1
	maxiter := 300
	crossoverType := OnePoint

	genetic := NewGenetic(chromosomeLength, uint32(populationSize), float32(perMutation), uint32(maxiter), crossoverType)

	bestChromosome, bestFitnesses := genetic.startLoop()

	fmt.Printf("Best Chromosome: %v, intersects: %v\n", bestChromosome.gens, bestChromosome.Intersects())
	bestChromosome.Print()

	fmt.Printf("Best Fitnesses: %v\n", bestFitnesses)
}
