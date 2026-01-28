package flconfig

import (
	"fmt"
	"github.com/AIoTwin-Adaptive-FL-Orch/fl-orchestrator/internal/common"
	"github.com/AIoTwin-Adaptive-FL-Orch/fl-orchestrator/internal/model"
)

type SimpleFlConfiguration struct {
	GlobalEpochs int32
	LocalEpochs  int32
	BatchSize    int32
	LearningRate float32
}

func NewSimpleFlConfiguration(globalEpochs int32, localEpochs int32, batchSize int32, learningRate float32) *SimpleFlConfiguration {
	return &SimpleFlConfiguration{
		GlobalEpochs: globalEpochs,
		LocalEpochs:  localEpochs,
		BatchSize:    batchSize,
		LearningRate: learningRate,
	}
}

func (config *SimpleFlConfiguration) GetOptimalConfiguration(nodes []*model.Node) *FlConfiguration {
	globalAggregator, _, clients := common.GetClientsAndAggregators(nodes)
	flGlobalAggregator := &model.FlAggregator{
		Id:              globalAggregator.Id,
		InternalAddress: fmt.Sprintf("%s:%s", "0.0.0.0", fmt.Sprint(common.GLOBAL_AGGREGATOR_PORT)),
		ExternalAddress: common.GetGlobalAggregatorExternalAddress(globalAggregator.Id),
		Port:            8080,
		NumClients:      int32(len(clients)),
		Rounds:          config.GlobalEpochs,
	}
	flClients := common.ClientNodesToFlClients(clients, flGlobalAggregator, config.LocalEpochs)
	return &FlConfiguration{
		GlobalAggregator: flGlobalAggregator,
		LocalAggregators: nil,
		Clients:          flClients,
		Epochs:           config.LocalEpochs,
		LocalRounds:      config.GlobalEpochs,
	}
}
