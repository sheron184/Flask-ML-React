import axios from 'axios';

const axiosApi = axios.create({
    baseURL: 'http://localhost:5000/api',
    headers: {
        'Access-Control-Allow-Origin': '*'
    },  
    withCredentials: true
});

axiosApi.defaults.headers.common['Access-Control-Allow-Origin'] = '*'; // Adjust this based on your server configuration
axiosApi.defaults.headers.common['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE'; // Adjust methods as needed
axiosApi.defaults.headers.common['Access-Control-Allow-Headers'] = 'Authorization, Content-Type';

export default axiosApi;