#pragma once

class TimerCapture : ens::ProgressBar {
private:
    std::vector<double>& loss_data_value;
    std::vector<double>& loss_data_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    std::chrono::duration<double> totalTime;
public:

    TimerCapture(std::vector<double>& loss_data_value,std::vector<double>& loss_data_time,const size_t widthIn = 70,std::ostream& output = arma::get_cout_stream()) 
    : ens::ProgressBar(widthIn,output), loss_data_value(loss_data_value),loss_data_time(loss_data_time){
        // do nothing
    }

    template<typename OptimizerType, typename FunctionType, typename MatType>
    bool EndEpoch(OptimizerType& opt,
    FunctionType& fn,
    const MatType& co,
    const size_t epoch,
    const double objective)
    {   
        this->endTime = std::chrono::high_resolution_clock::now();
        this->totalTime = std::chrono::duration<double>(this->endTime - this->startTime);
        this->loss_data_value.push_back(objective);
        this->loss_data_time.push_back(
            this->totalTime.count()
        );
        return false;
    }


};