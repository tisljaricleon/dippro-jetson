package florch

import (
	"fmt"
	"os"
	"strconv"

	"github.com/AIoTwin-Adaptive-FL-Orch/fl-orchestrator/internal/common"

	"github.com/AIoTwin-Adaptive-FL-Orch/fl-orchestrator/internal/model"
)

func BuildGlobalAggregatorConfigFiles(flAggregator *model.FlAggregator) (map[string]string, error) {
	configDirectoryPath := "../../configs/fl/"

	filesFolderPath := "../../files"
	filesFiles, err := common.ReadFolderToFileMap(filesFolderPath)
	if err != nil {
		return nil, fmt.Errorf("Failed to read files folder: %w", err)
	}

	// Add cloudlets.json from data folder
	cloudletsPath := "../../data/cloudlets.json"
	cloudletsBytes, err := os.ReadFile(cloudletsPath)
	var cloudletsString string
	if err == nil {
		cloudletsString = string(cloudletsBytes)
	} else {
		fmt.Printf("[BuildGlobalAggregatorConfigFiles] Warning: could not read cloudlets.json: %v\n", err)
	}

	taskBytesArray, err := os.ReadFile(fmt.Sprint(configDirectoryPath, "task/task.py"))
	if err != nil {
		fmt.Print(err)
	}
	taskString := string(taskBytesArray)

	// Add global_server.py from global_server directory
	globalServerPyPath := "../../internal/fl_service/global_server/global_server.py"
	globalServerPyBytes, err := os.ReadFile(globalServerPyPath)
	var globalServerPyString string
	if err == nil {
		globalServerPyString = string(globalServerPyBytes)
	} else {
		fmt.Printf("[BuildGlobalAggregatorConfigFiles] Warning: could not read global_server.py: %v\n", err)
	}

	globalAggregatorConfig := GlobalAggregatorConfig_Yaml

	filesData := map[string]string{
		"task.py":                   taskString,
		"global_server_config.yaml": globalAggregatorConfig,
	}

	if cloudletsString != "" {
		filesData["cloudlets.json"] = cloudletsString
	}

	if globalServerPyString != "" {
		filesData["global_server.py"] = globalServerPyString
	}

	for k, v := range filesFiles {
		fmt.Printf("[BuildClientConfigFiles] Adding file from files folder to client ConfigMap: %s\n", k)
		filesData[k] = v
	}

	return filesData, nil
}

func BuildLocalAggregatorConfigFiles(flAggregator *model.FlAggregator) (map[string]string, error) {
	localAggregatorConfig := fmt.Sprintf(LocalAggregatorConfig_Yaml, flAggregator.ParentAddress, strconv.Itoa(int(flAggregator.LocalRounds)))

	filesData := map[string]string{
		"local_server_config.yaml": localAggregatorConfig,
	}

	return filesData, nil
}

func BuildClientConfigFiles(client *model.FlClient) (map[string]string, error) {
	configDirectoryPath := "../../configs/fl/"

	filesFolderPath := "../../files"
	filesFiles, err := common.ReadFolderToFileMap(filesFolderPath)
	if err != nil {
		return nil, fmt.Errorf("Failed to read files folder: %w", err)
	}

	// Add cloudlets.json from data folder
	cloudletsPath := "../../data/cloudlets.json"
	cloudletsBytes, err := os.ReadFile(cloudletsPath)
	var cloudletsString string
	if err == nil {
		cloudletsString = string(cloudletsBytes)
	} else {
		fmt.Printf("[BuildClientConfigFiles] Warning: could not read cloudlets.json: %v\n", err)
	}

	taskBytesArray, err := os.ReadFile(fmt.Sprint(configDirectoryPath, "task/task.py"))
	if err != nil {
		fmt.Print(err)
	}
	taskString := string(taskBytesArray)

	// Add client.py from client_jetson directory
	clientPyPath := "../../internal/fl_service/client_jetson/client.py"
	clientPyBytes, err := os.ReadFile(clientPyPath)
	var clientPyString string
	if err == nil {
		clientPyString = string(clientPyBytes)
	} else {
		fmt.Printf("[BuildClientConfigFiles] Warning: could not read client.py: %v\n", err)
	}

	clientConfigString := fmt.Sprintf(ClientConfig_Yaml, client.ParentAddress, strconv.Itoa(int(client.PartitionId)),
		strconv.Itoa(int(client.NumPartitions)), strconv.Itoa(int(client.Epochs)), strconv.Itoa(int(client.BatchSize)),
		fmt.Sprintf("%f", client.LearningRate))

	filesData := map[string]string{
		"task.py":            taskString,
		"client_config.yaml": clientConfigString,
	}

	if cloudletsString != "" {
		filesData["cloudlets.json"] = cloudletsString
	}

	if clientPyString != "" {
		filesData["client.py"] = clientPyString
	}

	for k, v := range filesFiles {
		fmt.Printf("[BuildClientConfigFiles] Adding file from files folder to client ConfigMap: %s\n", k)
		filesData[k] = v
	}

	return filesData, nil
}

const GlobalAggregatorConfig_Yaml = `
server:
  address: "0.0.0.0:8080"
  global_rounds: 10

strategy:
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2
`

const LocalAggregatorConfig_Yaml = `
server:
  global_address: "%[1]s"
  local_address: "0.0.0.0:8080"
  local_rounds: %[2]s
  global_rounds: 1

strategy:
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2
`

const ClientConfig_Yaml = `
server:
  address: "%[1]s"

node_config:
  partition-id: %[2]s 
  num-partitions: %[3]s 

run_config:
  local-epochs: %[4]s 
  batch-size: %[5]s 
  learning-rate: %[6]s  
`
