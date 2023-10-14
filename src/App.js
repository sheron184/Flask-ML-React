import Button from '@mui/material/Button'
import axios from 'axios'
import { useEffect } from 'react'

function App() {
  useEffect(() => {
    axios({
      method: "GET",
      url:"/profile",
    })
    .then((response) => {
      const res =response.data
      console.log(res)
    }).catch((error) => {
      if (error.response) {
        console.log(error.response)
        console.log(error.response.status)
        console.log(error.response.headers)
        }
    })
  })
  return (
    <>
      <div>
          <Button variant='contained'>COL</Button>
      </div>
    </>
  )
}

export default App
