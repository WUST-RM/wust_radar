    }
    infer_status_.reserve(max_infer_threads_); 
    for (size_t i = 0; i < max_infer_threads_; ++i) {
        infer_status_.emplace_back(false);  
    }