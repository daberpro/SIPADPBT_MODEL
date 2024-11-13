/*
    PEMBERITAHUAN! kode - kode berikut ini
    dibuat dengan tujuan pelaksanaan CAPSTONE PROJECT
    dibuat pada tanggal 12/10/2024
    oleh Ari Susanto
*/
/*
    kode di bawah digunakan untuk mendefinisikan makro Training
    yang berfungsi untuk memuat proses training pada model
    jika ingin meload model yang sudah disimpan
    silahkan comment kode di bawah
*/
// #define TRAINING

/*  uncomment kode di bawah untuk mendefinisikan makro JENGKOL jika ingin memuat pelatihan 
    untuk data jengkol, jika ingin memuat pelatihan untuk
    durian abaikan saja kode di bawah
*/
#define JENGKOL

#define MLPACK_ENABLE_ANN_SERIALIZATION
#define CEREAL_THREAD_SAFE 1


#include <mlpack.hpp>
#include "TimerCapture.h"

int main(){
    // mendapatkan path lokasi aplikasi 
    // berjalan dan konversi ke string
    std::string current_path = std::filesystem::current_path().string();

    // mengambil index data penyakit
    std::ifstream index_data_file(current_path + "/index data.json");
    nlohmann::json disease_index = nlohmann::json::parse(index_data_file);

    // membuat matrix sebagai container data
    arma::mat data;
    bool is_data_error = false;

    #ifdef TRAINING
    
    mlpack::data::Load(
        #ifdef JENGKOL
        current_path+"/data jengkol.csv",
        #else
        current_path+"/data durian.csv",
        #endif
        data,
        is_data_error,
        false,
        mlpack::data::FileType::CSVASCII
    ); // memuat data dari file CSV dan menyimpan ke variabel data


    #ifdef JENGKOL
    std::cout << "data path \"" << (current_path+"/data jengkol.csv\"\n");
    #else
    std::cout << "data path \"" << (current_path+"/data durian.csv\"\n");
    #endif
    if(is_data_error){
        std::cout << "error when get data\n";
    }

    #ifdef JENGKOL
    arma::mat targets = data.cols(data.n_cols - 11, data.n_cols-1);
    arma::mat inputs = data.cols(0, data.n_cols - 12);
    targets = targets.t();
    inputs = inputs.t();
    std::cout << "Inputs \n" << inputs << "\n";
    std::cout << "Targets \n" << targets << "\n";
    #else
    arma::mat targets = data.cols(data.n_cols - 9, data.n_cols-1);
    arma::mat inputs = data.cols(0, data.n_cols - 10);
    targets = targets.t();
    inputs = inputs.t();
    std::cout << "Inputs \n" << inputs << "\n";
    std::cout << "Targets \n" << targets << "\n";
    #endif

    /*
        model akan dibuat menggunakan Feed Forward Network (FFN)
        dengan arsitektur sebagai berikut
        21 -> 21 -> 21 -> 11 JENGKOL
        16 -> 16 -> 10 -> 9  DURIAN

        dengan perincian JENGKOL sebagai berikut:
        21 Input
        21 Hidden Layer 1
        21 Hidden Layer 2
        11 Ouput

        dengan perincian DURIAN sebagai berikut:
        16 Input
        16 Hidden Layer 1
        10 Hidden Layer 2
        9 Ouput

        model akan menggunakan fungsi aktivasi sigmoid dengan loss function
        cross-entropy dan aktivasi output meggunakan soft-max
    */
    mlpack::FFN<mlpack::CrossEntropyError> model;

    #ifdef JENGKOL
    // model.Add<mlpack::Linear>(21);
    model.Add<mlpack::Sigmoid>(); // 21 Input
    model.Add<mlpack::Linear>(21); 
    model.Add<mlpack::Sigmoid>(); // 21 Hidden Layer 1
    model.Add<mlpack::Linear>(21); 
    model.Add<mlpack::Sigmoid>(); // 21 Hidden Layer 2
    model.Add<mlpack::Linear>(11); 
    model.Add<mlpack::Softmax>(); // 11 Output
    #else
    // model.Add<mlpack::Linear>(16); 
    model.Add<mlpack::Sigmoid>(); // 16 Input
    model.Add<mlpack::Linear>(16); 
    model.Add<mlpack::Sigmoid>(); // 16 Hidden Layer 1
    model.Add<mlpack::Linear>(10); 
    model.Add<mlpack::Sigmoid>(); // 10 Hidden Layer 2
    model.Add<mlpack::Linear>(9); 
    model.Add<mlpack::Softmax>(); // 9 Output
    #endif

    // melakukan pelatihan pada model
    // dengan menampilkan loss selama proses pelatihan
    std::vector<double> loss_data_value;
    std::vector<double> loss_data_time;
    TimerCapture lossInTime(loss_data_value,loss_data_time);
    model.Train(
        inputs,
        targets,
        ens::ProgressBar(),
        ens::PrintLoss(),
        ens::Report(),
        lossInTime
    );
    matplot::plot(loss_data_time, loss_data_value, "-o")->line_width(3);
    matplot::title("Training Loss/Times (sec)");
    matplot::xlabel("Time in sec");
    matplot::ylabel("loss");
    matplot::show();

    #ifdef JENGKOL
    arma::mat testing("\
        1	1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	0	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	1	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0\
        0	1	0	0	1	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0\
        1	0	0	0	0	0	0	0	0	1	0	0	1	1	0	0	0	0	0	0	0\
        1	0	1	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0\
        0	0	1	0	0	1	0	0	0	1	0	0	0	0	0	0	1	1	0	0	0\
        1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0\
        1	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	1\
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
    ");
    testing.reshape(21,targets.row(0).n_cols);
    // menyimpan parameter - parameter
    // hasil pelatihan model ke dalam file
    mlpack::data::Save("model-jengkol.bin","model",model,false);
    #else
    arma::mat testing("\
        1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0\
        1	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0\
        0	0	0	0	0	0	1	1	1	0	0	0	0	0	0	0\
        1	1	0	0	0	0	0	0	0	1	0	0	0	0	0	0\
        0	0	0	0	0	0	0	0	0	0	1	1	1	0	0	0\
        1	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0\
        0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	0\
        1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	1\
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
    ");
    testing.reshape(16,targets.row(0).n_cols);
    // menyimpan parameter - parameter
    // hasil pelatihan model ke dalam file
    mlpack::data::Save("model-durian.bin","model",model,false);
    #endif

    arma::mat result_testing;
    model.Predict(testing,result_testing);
    std::cout << "Testing \n" << testing << "\n";
    std::cout << "Result Testing \n" << result_testing << "\n";
    
    #else

    mlpack::FFN<mlpack::CrossEntropyError> model;
    mlpack::data::Load(
        #ifdef JENGKOL
        current_path + "/model-jengkol.bin",
        #else
        current_path + "/model-durian.bin",
        #endif
        "model",
        model,
        false
    );

    
    mlpack::data::Load(
        #ifdef JENGKOL
        current_path+"/data jengkol.csv",
        #else
        current_path+"/data durian.csv",
        #endif
        data,
        is_data_error,
        false,
        mlpack::data::FileType::CSVASCII
    ); // memuat data dari file CSV dan menyimpan ke variabel data

    #ifdef JENGKOL
    arma::mat targets = data.cols(data.n_cols - 11, data.n_cols-1);
    arma::mat inputs = data.cols(0, data.n_cols - 12);
    targets = targets.t();
    inputs = inputs.t();
    std::cout << "Inputs \n" << inputs << "\n";
    std::cout << "Targets \n" << targets << "\n";
    #else
    arma::mat targets = data.cols(data.n_cols - 9, data.n_cols-1);
    arma::mat inputs = data.cols(0, data.n_cols - 10);
    targets = targets.t();
    inputs = inputs.t();
    std::cout << "Inputs \n" << inputs << "\n";
    std::cout << "Targets \n" << targets << "\n";
    #endif

    #ifdef JENGKOL
    arma::mat testing("\
        1	1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	0	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	1	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0	0	0\
        0	1	0	0	1	0	0	0	0	0	1	1	0	0	0	0	0	0	0	0	0\
        1	0	0	0	0	0	0	0	0	1	0	0	1	1	0	0	0	0	0	0	0\
        1	0	1	0	0	0	0	0	0	0	0	0	0	0	1	1	0	0	0	0	0\
        0	0	1	0	0	1	0	0	0	1	0	0	0	0	0	0	1	1	0	0	0\
        1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0\
        1	1	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
        0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	1\
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
    ");
    testing.reshape(21,targets.row(0).n_cols);
    // menyimpan parameter - parameter
    // hasil pelatihan model ke dalam file
    mlpack::data::Save("model-jengkol.bin","model",model,false);
    #else
    arma::mat testing("\
        1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	0\
        1	0	0	1	1	1	0	0	0	0	0	0	0	0	0	0\
        0	0	0	0	0	0	1	1	1	0	0	0	0	0	0	0\
        1	1	0	0	0	0	0	0	0	1	0	0	0	0	0	0\
        0	0	0	0	0	0	0	0	0	0	1	1	1	0	0	0\
        1	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0\
        0	1	0	0	0	0	0	0	0	0	0	0	0	0	1	0\
        1	0	0	1	0	0	0	0	0	0	0	0	0	0	0	1\
        0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0\
    ");
    testing.reshape(16,targets.row(0).n_cols);
    // menyimpan parameter - parameter
    // hasil pelatihan model ke dalam file
    mlpack::data::Save("model-durian.bin","model",model,false);
    #endif

    arma::mat result_testing;
    model.Predict(testing,result_testing);
    std::cout << "Testing \n" << testing << "\n";
    std::cout << "Result Testing \n" << result_testing << "\n";
    

    arma::mat input;
    arma::mat output;
    #ifdef JENGKOL
    input.reshape(1,21);
    #else
    input.reshape(1,16);
    #endif
    
    uint32_t colWidth = 50;
    uint32_t cli_input, index = 0;
    for(
        auto& element: 
        #ifdef JENGKOL
        disease_index["jengkol"]["gejala"]
        #else
        disease_index["durian"]["gejala"]    
        #endif
    ){
        std::cout << std::left 
        << std::setw(colWidth) << element.template get<std::string>()
        << " : ";
        std::cin >> cli_input;
        std::cin.ignore();
        input(0,index) = cli_input;
        index++;
    }

    input = input.t();
    model.Predict(input,output);

    std::cout << "\nResult Prediction \n";
    std::cout << "==========================\n";
    for(uint32_t i = 0; i < 11; i++){
        std::cout << std::left << std::fixed << std::setprecision(10) 
        << std::setw(colWidth)
        #ifdef JENGKOL 
        << disease_index["jengkol"]["penyakit"][i].template get<std::string>()
        #else
        << disease_index["durian"]["penyakit"][i].template get<std::string>()
        #endif
        << " : " << output(i,0) << "\n";
    }
    
    #endif


    return EXIT_SUCCESS;
}