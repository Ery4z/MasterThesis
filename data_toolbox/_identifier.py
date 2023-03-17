from .analysis_utilities import MultimodalAnalysisResult
import uuid
import numpy as np

class VehicleIdentifier:
    def __init__(self,decay_duration_identifier=60):
        
        self.__activated_metrics = []
        self.__activated_metrics_weight = []
        self.__activated_metrics_name = []
        
        self.__identifier_pool = {}
        
        self.__decay_duration_identifier = decay_duration_identifier
        
        
        
        
    
    def identify(self,analysis_result:MultimodalAnalysisResult,get_score_detail=False):
        """Main function to identify and labelise the vehicle in a image.
        This function is not idempotent and rely on previous sequencial call.

        Args:
            analysis_result (MultimodalAnalysisResult): Result of the analysis.
        """
        metrics_result = []
        """
        {
            "new": score_float,
            "<vehicle_id>": score_float,
            ...
        }
        
        """
        # This is to clean the non relevant identifiers.
        # This is before the analysis as some identifiers may have decayed since lot of time if the last image identified has a great timestamp difference.
        self.__clean_up(analysis_result)
    
    
        # Calculate each metric
    
        for metrics in self.__activated_metrics:
            m_result = metrics(analysis_result)
            
            metrics_result.append(m_result)
            
        # Aggregate each result
        aggregate = {}
        
        for i,result in enumerate(metrics_result):
            for identifier,value in result.items():
                if identifier not in aggregate:
                    aggregate[identifier] = [0,0]
                
                
                # Weighted average
                aggregate[identifier][0] += value*self.__activated_metrics_weight[i]
                aggregate[identifier][1] += self.__activated_metrics_weight[i]
                
        ranking_list = []
        
        # How many metrics are required to find a new vehicle
        new_vehicle_threshold = 1
        
        aggregate["new"] = [aggregate["new"][0]+new_vehicle_threshold,aggregate["new"][1]]
        
        for identifier, value in aggregate.items():
                
            if value[1] == 0:
                continue
            ranking_list.append([identifier,value[0]/value[1]])
            
            
        
        ranking_list = sorted(ranking_list, key=lambda a: a[1] )
        
        
        # Create a list of the insight of the score, each metric and its value for the selected vehicle.
        # Create a list for each metric with the following format: ["metric_name",metric_value_for_top_existing_id,metric_value_for_new]
        score_detail = []
        if get_score_detail:
            for metric_name,metric_result in zip(self.__activated_metrics_name,metrics_result):
                
                if ranking_list[0][0] == "new":
                    if len(ranking_list) > 1:
                        score_detail.append([metric_name,round(metric_result[ranking_list[1][0]],3),round(metric_result["new"],3)])
                    else:
                        score_detail.append([metric_name,"/",round(metric_result["new"],3)])
                else:
                    score_detail.append([metric_name,round(metric_result[ranking_list[0][0]],3),round(metric_result["new"],3)])
        
            # Add the total score
            if ranking_list[0][0] == "new":
                if len(ranking_list) > 1:
                    score_detail.append(["total",round(ranking_list[1][1],3),round(ranking_list[0][1],3)])
                else:
                    score_detail.append(["total","/",round(ranking_list[0][1],3)])
            else:
                score_new = 0
                for i in range(len(ranking_list)):
                    if ranking_list[i][0] == "new":
                        score_new = ranking_list[i][1]
                        break
                score_detail.append(["total",round(ranking_list[0][1],3),round(score_new,3)])
                
        
        if ranking_list[0][0] == "new":
            vehicle_identifier = uuid.uuid4()
        else :
            try:
                vehicle_identifier = ranking_list[0][0]
            except IndexError:
                vehicle_identifier = None
        
        if vehicle_identifier is not None:
            self.__identifier_pool[vehicle_identifier] = analysis_result
        
        
        
        
        if get_score_detail:
            return vehicle_identifier,score_detail
        return vehicle_identifier
        
    def __clean_up(self,analysis_result):
        to_delete = []
        for identifier in self.__identifier_pool:
            last_identified_data: MultimodalAnalysisResult  = self.__identifier_pool[identifier]
            
            timestamp_delta = float(analysis_result.timestamp) - float(last_identified_data.timestamp)
            if timestamp_delta > self.__decay_duration_identifier:
                to_delete.append(identifier)
        for identifier in to_delete:
            del self.__identifier_pool[identifier]
                
        
    def __metric_example(self,analysis_result:MultimodalAnalysisResult):
        """This metric has for purpose to show a basic metric creation.
        This metric calculate the radar distance supposed by prediction using the speed and difference in timestamp.
        It set the new id to 5 as any error of more than 5 meters would be suspect.

        Args:
            analysis_result (MultimodalAnalysisResult): _description_
        """
        
        identifier_pool = self.__identifier_pool
        
        score = {
            "new": 5,
        }
        
        for identifier in identifier_pool:
            last_identified_data: MultimodalAnalysisResult  = identifier_pool[identifier]
            
            timestamp_delta = analysis_result.timestamp - last_identified_data.timestamp
            
            previous_distance = last_identified_data.heatmap_info[0]["distance"]
            previous_speed = last_identified_data.heatmap_info[0]["speed"]
            
            current_distance = analysis_result.heatmap_info[0]["distance"]
            
            predicted_distance = previous_distance +  previous_speed*timestamp_delta/3.6
            
            score[identifier] = abs(predicted_distance - current_distance)
        
        # Normalize the score
        maximum = max(score.values())
        for identifier in score:
            score[identifier] = score[identifier]/maximum
        
        
        return score
            
    def set_metric_example(self,weight=1):
        self.__activated_metrics.append(self.__metric_example)
        self.__activated_metrics_weight.append(weight)
        self.__activated_metrics_name.append("rdr_pred")
        
        return self
            
    def __metric_euclidian_distance(self,analysis_result:MultimodalAnalysisResult):
        """The purpose of this metric is to calculate the 3D euclidian distance between position of the vehicle in the image and the previous position.
        """
        
        identifier_pool = self.__identifier_pool
        
        score = {
            "new": 5,
        }
        
        for identifier in identifier_pool:
            last_identified_data: MultimodalAnalysisResult  = identifier_pool[identifier]
            
            timestamp_delta = analysis_result.timestamp - last_identified_data.timestamp
            
            previous_position = np.array(last_identified_data.position)
            current_position = np.array(analysis_result.position)
            
            delta = current_position - previous_position

            # Euclidian distance
            score[identifier] = np.sqrt(np.sum(np.square(delta)))
        
        # Normalize the score
        maximum = max(score.values())
        if maximum != 0:
            for identifier in score:
                score[identifier] = score[identifier]/maximum
        
        
        
        return score
    
    def set_metric_euclidian_distance(self,weight=1):
        self.__activated_metrics.append(self.__metric_euclidian_distance)
        self.__activated_metrics_weight.append(weight)
        self.__activated_metrics_name.append("eucl_dist")
        
        return self
    
    def __metric_bb_common_area(self,analysis_result:MultimodalAnalysisResult):
        """The purpose of this metric is to calculate the 3D euclidian distance between position of the vehicle in the image and the previous position.
        """
        # TODO: THIS
        
        identifier_pool = self.__identifier_pool
        
        # For needing at least 1 pixel in common to be considered as the same vehicle
        score = {
            "new": 10,
        }
        
        for identifier in identifier_pool:
            last_identified_data: MultimodalAnalysisResult  = identifier_pool[identifier]
            
            timestamp_delta = analysis_result.timestamp - last_identified_data.timestamp
            
            previous_bb = np.array(last_identified_data.image_info[0]["bbox"])
            current_bb = np.array(analysis_result.image_info[0]["bbox"])
            
            common_area =  np.array([max(previous_bb[0],current_bb[0]),max(previous_bb[1],current_bb[1]),min(previous_bb[2],current_bb[2]),min(previous_bb[3],current_bb[3])])
            

            
            area_int = (max(common_area[2]-common_area[0],0))*(max(common_area[3]-common_area[1],0))
            # The more area the lower the score
            
            
            # Euclidian distance
            score[identifier] = area_int
        
        # Normalize the score
        maximum = max(score.values())
        if maximum != 0:
            for identifier in score:
                score[identifier] = score[identifier]/maximum
                
        # Reversing the score 0 -> 1 and 1 -> 0
        
        for identifier in score:
            score[identifier] = 1 - score[identifier]
        
        # Fixing the new to 0 as this metric should not penalise new score
        score["new"] = 0
        
        
        
        return score
    
    def set_metric_bb_common_area(self,weight=2):
        self.__activated_metrics.append(self.__metric_bb_common_area)
        self.__activated_metrics_weight.append(weight)
        self.__activated_metrics_name.append("bb_cmn_area")
        
        return self